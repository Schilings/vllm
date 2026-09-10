# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import hashlib
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NewType, TypeAlias, cast, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None
    # Number of prefix tokens covered by _block_hash. For full blocks this is
    # the full block boundary; partial aliases can end inside a cache block.
    _block_hash_num_tokens: int | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        return self._block_hash_num_tokens

    def set_block_hash(
        self,
        block_hash: BlockHashWithGroupId,
        num_tokens: int | None = None,
    ) -> None:
        assert self.block_hash is None and self._block_hash_num_tokens is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None
        self._block_hash_num_tokens = None

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"_block_hash_num_tokens={self._block_hash_num_tokens}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks at the front of the free list."""
        if len(blocks) == 0:
            return

        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, (
            "next_free_block of fake_free_list_head should always exist"
        )

        prev_block = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block

        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block

        self.num_free_blocks += len(blocks)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
        request (Request): The request.

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys.
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA name.
    # Request with provided cache salt need to include the salt.
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx >= offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input. Include its offset
            # relative to the start of the block so prefix-cache keys stay
            # distinct when the same MM item appears at different positions
            # within otherwise-identical placeholder blocks.
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """Generate extra keys related to prompt embeds for block hash computation.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.

    Returns:
        Return a stable hash of the block prompt embeddings if prompt embeds
        are present. Return empty list otherwise.
    """
    if request.prompt_embeds is None:
        return []
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # Hash prompt embeds once per block and cache on request
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler (e.g. for ``num_computed_tokens`` rounding). Single group:
      ``cache_config.block_size * dcp * pcp``. Multiple groups: LCM of every
      group's block size — context parallelism is not supported here.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.hash_block_size`` override if set, else the GCD of group
      block sizes; every group's block size must be divisible by it. Returns
      the scheduler block size (i.e. disables finer hashing) if block hashing
      is inactive or a mamba group's block size diverges from the cache
      block size (mamba_cache_mode != "align").
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:  # Single group: block_size * dcp * pcp
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        raise ValueError(
            "Hybrid KV cache groups with multiple block sizes do not "
            "support context parallelism (dcp_world_size/pcp_world_size > 1)."
        )

    group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
    scheduler_block_size = math.lcm(*group_block_sizes)

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # Mamba groups with block_size != cache_config.block_size
    # (mamba_cache_mode != "align") break divisibility; back off to the
    # scheduler block size.
    if any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.block_size != cache_config.block_size
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size

    requested = cache_config.hash_block_size
    hash_block_size = (
        requested if requested is not None else math.gcd(*group_block_sizes)
    )
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid hash_block_size={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by hash_block_size. "
            f"Got group block sizes={group_block_sizes}."
        )
    return scheduler_block_size, hash_block_size


def get_request_block_hasher(
    hash_block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request.

    Hashes are computed at ``hash_block_size`` granularity and chained over the
    full prefix, so each hash uniquely fingerprints the prefix ending at its
    boundary. Coarser group block sizes and partial-cache boundaries reuse
    these hashes directly (see ``BlockHashListWithBlockSize``).
    """

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * hash_block_size
        num_tokens = request.num_tokens

        if start_token_idx + hash_block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + hash_block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += hash_block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.
    """
    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config, (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups)
    )
    memory_per_block = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        * num_layer_per_group
    )
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


def _pool_bytes_per_block(
    vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]
) -> int:
    """
    Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes
    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # buckets = {page_size: [[layer_names], [layer_names], ...]}
        buckets = _bucket_layers_by_page_size(kv_cache_groups)
        return sum(ps * len(slots) for ps, slots in buckets.items())
    group_size = max(len(g.layer_names) for g in kv_cache_groups)
    page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
    return page_size * group_size


def get_num_blocks(
    vllm_config: VllmConfig,
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    return may_override_num_blocks(vllm_config, num_blocks)


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    # list(kv_cache_specs.keys()) => 所有layer分到一个group里
    # 返回一个 [ KVCacheGroupSpec ]
    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same page size.
    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same page size, False otherwise.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, first try to
    unify page size by increasing the block size of layers with smaller page
    size. If a smaller attention page does not evenly divide the maximum page
    size, keep its logical block size and pad its physical page instead --- but
    only for attention layers whose backend opts in via
    ``AttentionSpec.indexes_kv_by_block_stride`` (the padded page is read through
    a strided view, which not every backend handles). Raise NotImplementedError
    if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            # ⚠️ 如果page size不一致
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif (
                isinstance(layer_spec, AttentionSpec)
                and layer_spec.indexes_kv_by_block_stride
            ):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for attention layers whose backend indexes KV "
                    "pages by the block stride (indexes_kv_by_block_stride is "
                    "True)."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def _bucket_layers_by_page_size(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> dict[int, list[list[str]]]:
    """Bucket layers by page size: ``result[ps][slot_idx] = [layer_names]``.

    Layers from different groups at the same ``slot_idx`` share an underlying tensor
    (they have independent block tables so block-id namespaces never collide).
    """
    """
    1️⃣假设有 2 个 group，layer 的 page_size 如下：
        Group 0:  L0(大=100)  L1(小=20)  L2(大=100)
                   ↑           ↑         ↑
        Group 1:  L3(大=100)  L4(小=20)  L5(大=100)  L6(小=20)
    2️⃣buckets = {
                100: [ [L0, L3],     # slot 0：两个 group 的"第1个大页 layer"
                       [L2, L5] ],   # slot 1：两个 group 的"第2个大页 layer"
                20: [ [L1, L4],     # slot 0：两个 group 的"第1个小页 layer"
                      [L6]     ],   # slot 1：只有 Group1 有第2个小页 layer
              }
    3️⃣一个物理 block 的内存布局（block_stride = 各桶尺寸之和）：
     ┌──────────100─────────┬──────────100─────────┬────20───┬────20───┐
     │  buckets[100][0]     │  buckets[100][1]     │ [20][0] │ [20][1] │
     │  L0/L3 共享           │  L2/L5 共享           │ L1/L4   │  L6     │
     └──────────────────────┴──────────────────────┴─────────┴─────────┘
     offset=0               offset=100             offset=200 offset=220

    [L0, L3] 为什么能挤进同一段内存而不冲突？
    因为所有 group 共用同一个全局 BlockPool，block-id 全局唯一：任一 id
    同一时刻只属于一个 group（总需求 = 各组之和，见 coordinator 的
    get_num_blocks_to_allocate）。L0(Group0) 只写自己领到的那些行，
    L3(Group1) 只写另一批行——同一张量、行集合互斥，物理上永不相撞。
    反过来，同一个 group 内的两个大页 layer（L0、L2）不能共享同一 slot：
    它们共用一张 block table，同一个 block-id 对两层意味着同一段字节，
    会互相覆盖，所以被排到 slot 0 和 slot 1 两个不同的桶。

    ============================================================
    ⚠️ DeepSeek-V4 真实示例（这才是 packed 布局的主力场景）
    ============================================================
    区别: DeepSeek-V4 只有 2 个 group, 但每个 group 是
    UniformTypeKVCacheSpecs —— 同一个 group 内部各 layer 的 page_size 不同
    取 n_c4=2, n_c128=1, n_swa=3 (n_swa = n_c4 + n_c128,
    SWA 组层数与 MLA 组的 c4+c128 层数一一对应; page_size 单位: 字节):
        Group 0 (mla_group):
            c4_mla.0 (37440)  c4_idx.0 (8640)
            c4_mla.1 (37440)  c4_idx.1 (8640)
            c128_mla.0 (1728)
        Group 1 (swa_group):
            swa.0 (37440)     swa.1 (37440)     swa.2 (37440)

    最终 buckets:
        37440: [ [c4_mla.0, swa.0],    # slot0: 跨 group 共享
                 [c4_mla.1, swa.1],    # slot1: 跨 group 共享
                 [swa.2] ]             # slot2: 只有 Group1 (第3个大页)
         8640: [ [c4_idx.0],           # slot0: 只有 Group0
                 [c4_idx.1] ]          # slot1: 只有 Group0
         1728: [ [c128_mla.0] ]        # slot0: 只有 Group0

        ⚠️ 一个物理 block 的内存布局 ==> 放了所有layer同个block的kv cache
        (block_stride = 37440*3 + 8640*2 + 1728 = 131328):
         ┌────37440─────┬────37440─────┬───37440─────┬──8640───┬──8640───┬─1728─┐
         │ [37440][0]   │ [37440][1]   │[37440][2]   │[8640][0]│[8640][1]│[1728]│
         │c4_mla.0/swa.0│c4_mla.1/swa.1│ swa.2       │c4_idx.0 │c4_idx.1 │c128.0│
         └──────────────┴──────────────┴─────────────┴─────────┴─────────┴──────┘
         off=0          off=37440       off=74880    off=112320 off=120960 off=129600
        
         ⚠️ 这里是 c4_mla.0/swa.0 是 或 的关系，不是 和 的关系！！！！！！ 
    """
    # ⚠️ result[page_size][slot_idx] = [能共用同一块底层张量的 layer 名单]
    buckets: dict[int, list[list[str]]] = defaultdict(list)
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        # ⚠️ 统计这个group的不同layer的采取的page size的频数
        slot_count: dict[int, int] = defaultdict(int)
        for layer_name in group.layer_names:
            if isinstance(spec, UniformTypeKVCacheSpecs):
                ps = spec.kv_cache_specs[layer_name].page_size_bytes
            else:
                # 如果是正常hybrid，每层的page size完全一致
                ps = spec.page_size_bytes
            slot_idx = slot_count[ps]
            slot_count[ps] += 1
            # ⚠️ ！！
            if slot_idx == len(buckets[ps]):
                buckets[ps].append([])
            #
            buckets[ps][slot_idx].append(layer_name)
    return buckets


def _use_packed_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    is_dsv4 = all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    )
    kv_transfer_config = vllm_config.kv_transfer_config
    extra_config = (
        kv_transfer_config.kv_connector_extra_config
        if kv_transfer_config is not None
        else {}
    )
    # NOTE: enable_cross_layers_blocks is an experimental API and subject to change with
    # https://github.com/vllm-project/vllm/issues/42082
    enable_cross_layers = (
        str(extra_config.get("enable_cross_layers_blocks", "False")).lower() == "true"
    )
    return is_dsv4 or (enable_cross_layers and len(kv_cache_groups) > 1)


def _get_kv_cache_config_packed(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """Plan a packed per-block KV cache tensor layout.

    Emit one KVCacheTensor per (slot_idx, page_size). Layers from different
    groups at the same slot share a tensor (they have independent block
    tables so block-id namespaces never collide). Each emitted tensor aliases
    one physical backing allocation, with per-block data laid out contiguously.
    """
    """
        1️⃣假设有 2 个 group，layer 的 page_size 如下：
            Group 0:  L0(大=100)  L1(小=20)  L2(大=100)
                       ↑           ↑         ↑
            Group 1:  L3(大=100)  L4(小=20)  L5(大=100)  L6(小=20)
        2️⃣buckets = {
                    100: [ [L0, L3],     # slot 0：两个 group 的"第1个大页 layer"
                           [L2, L5] ],   # slot 1：两个 group 的"第2个大页 layer"
                    20: [ [L1, L4],     # slot 0：两个 group 的"第1个小页 layer"
                          [L6]     ],   # slot 1：只有 Group1 有第2个小页 layer
                  }
        3️⃣一个物理 block 的内存布局（block_stride = 各桶尺寸之和）：
         ┌─────────100x2────────┬─────────100x2────────┬───20x2──┬────20───┐
         │  buckets[100][0]     │  buckets[100][1]     │ [20][0] │ [20][1] │
         │  L0/L3 共享           │  L2/L5 共享           │ L1/L4   │  L6     │
         └──────────────────────┴──────────────────────┴─────────┴─────────┘
         offset=0

        [L0, L3] 为什么能挤进同一段内存而不冲突？
        因为它们来自不同 group，各自有独立的 block table（独立的 block-id 命名空间）。
        同一时刻，Group0 用 block-id=5、Group1 也用 block-id=5，但它们指向的是不同物理 block——block table 映射不同，永不撞车。
        反过来，同一个 group 内的两个大页 layer（L0、L2）不能共享，因为它们共用一张 block table，block-id=5 对两者是同一块，会互相覆盖
        所以它们被分到 slot 0 和 slot 1 两个不同的桶。

        ============================================================
        ⚠️ DeepSeek-V4 真实示例（这才是 packed 布局的主力场景）
        ============================================================
        区别: DeepSeek-V4 只有 2 个 group, 但每个 group 是
        UniformTypeKVCacheSpecs —— 同一个 group 内部各 layer 的 page_size 不同
        取 n_c4=2, n_c128=1, n_swa=3 (n_swa = n_c4 + n_c128,
        SWA 组层数与 MLA 组的 c4+c128 层数一一对应; page_size 单位: 字节):
            Group 0 (mla_group):
                c4_mla.0 (37440)  c4_idx.0 (8640)
                c4_mla.1 (37440)  c4_idx.1 (8640)
                c128_mla.0 (1728)
            Group 1 (swa_group):
                swa.0 (37440)     swa.1 (37440)     swa.2 (37440)

        最终 buckets:
            37440: [ [c4_mla.0, swa.0],    # slot0: 跨 group 共享
                     [c4_mla.1, swa.1],    # slot1: 跨 group 共享
                     [swa.2] ]             # slot2: 只有 Group1 (第3个大页)
             8640: [ [c4_idx.0],           # slot0: 只有 Group0
                     [c4_idx.1] ]          # slot1: 只有 Group0
             1728: [ [c128_mla.0] ]        # slot0: 只有 Group0

        ⚠️ 一个物理 block 的内存布局 ==> 放了所有layer同个block的kv cache
        (block_stride = 37440*3 + 8640*2 + 1728 = 131328):
         ┌────37440─────┬────37440─────┬───37440─────┬──8640───┬──8640───┬─1728─┐
         │ [37440][0]   │ [37440][1]   │[37440][2]   │[8640][0]│[8640][1]│[1728]│
         │c4_mla.0/swa.0│c4_mla.1/swa.1│ swa.2       │c4_idx.0 │c4_idx.1 │c128.0│
         └──────────────┴──────────────┴─────────────┴─────────┴─────────┴──────┘
         off=0          off=37440       off=74880    off=112320 off=120960 off=129600
        
         ⚠️ 这里是 c4_mla.0/swa.0 是 或 的关系，不是 和 的关系！！！！！！
    """
    # buckets = {page_size: [[layer_names], [layer_names], ...]}
    buckets = _bucket_layers_by_page_size(kv_cache_groups)
    total_num_bytes_per_block = sum(ps * len(slots) for ps, slots in buckets.items())
    # ⚠️ 一个物理 block 的内存布局 ==> 放了所有layer同个block的kv cache
    # ⚠️⚠️⚠️ 这种设计就是： 同样的block，可以通用，不同group的kv可以拿去存储！！！！
    # ⚠️⚠️⚠️ 例如dsv4，两种kv group( Uniform(CSA+CIA+HCA), Uniform(SWA) ) 都可以直接拿这种 block 进行存储
    # ⚠️⚠️⚠️ 缺点就是：可能每个block都存在显存浪费！！！！
    num_blocks = available_memory // total_num_bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    total_size = total_num_bytes_per_block * num_blocks

    # ⚠️ 整个kv cache放在一个大Tensor里，但是要告诉每个group的每个layer各自怎么访问
    kv_cache_tensors: list[KVCacheTensor] = []
    byte_offset = 0
    for ps, slots in buckets.items():
        for slot in slots:
            kv_cache_tensors.append(
                KVCacheTensor(
                    # ⚠️ size都是一样的！
                    # ⚠️ 通过 block_id * total_num_bytes_per_block + offset来访问自己的部分
                    size=total_size,
                    # ⚠️ slot0 = [c4_mla.0, swa.0]、slot1 = [c4_mla.1, swa.1], slot2 = [swa.2]
                    shared_by=slot,
                    offset=byte_offset,
                    block_stride=total_num_bytes_per_block,
                )
            )
            byte_offset += ps

    return num_blocks, kv_cache_tensors


_get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_packed


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # ⚠️ 从 groups 构建 KVCacheConfig(num_blocks, kv_cache_tensors)。
    # 三种布局分支: A) 单 uniform group  B) packed(DSv4)  C) general 混合。
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden sizes. Allocate different amount of memory for each
        # layer based on its hidden size.
        # 1️⃣ 单 uniform group：不同layer的page_size_bytes可能不同。
        #   因此将单个block的所有layer的page融合在一个block里，page_size_bytes = sum(layer's page_size_bytes)
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 uses the packed layout by default. Other multi-group
        # layouts can opt in with --enable-cross-layers.
        # 2️⃣ DSv4 or 开启packed( enable_cross_layers_blocks )
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(
            vllm_config, kv_cache_groups, available_memory
        )

    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2

        # 3️⃣ general 混合。
        group_size = max(len(group.layer_names) for group in kv_cache_groups)
        # ⚠️ 强制所有group的page size完全一致！
        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_swa_mla = any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    )

    uniform_block_size: int | None = None
    if has_swa_mla:
        # For DeepseekV4, block sizes can be different for different KV cache groups.
        # E.g., Full MLA: 256; SWA MLA: 64; C4 partial states: 4, C128 states: 8.
        assert has_full_attention
        any_full_spec = next(
            iter(
                spec
                for spec in kv_cache_spec.values()
                if isinstance(spec, FullAttentionSpec)
            )
        )
        uniform_block_size = any_full_spec.block_size

    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowMLASpec):
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=uniform_block_size
                    if uniform_block_size is not None
                    else spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    page_size_padded=spec.page_size_padded,
                    cache_dtype_str=spec.cache_dtype_str,
                    alignment=spec.alignment,
                    compress_ratio=spec.compress_ratio,
                    model_version=spec.model_version,
                )
            elif isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    head_size_v=spec.head_size_v,
                    dtype=spec.dtype,
                    kv_quant_mode=spec.kv_quant_mode,
                    sliding_window=spec.sliding_window,
                    page_size_padded=spec.page_size_padded,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=spec.page_size_padded,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    将 KV cache spec 按注意力类型分组，每组合并为一个 UniformTypeKVCacheSpecs。
    目前仅用于 DeepSeekV4。

    DeepSeekV4 一个 transformer block 内同时注册了多类 KV cache 层（compress_ratio
    不同，key 的数量也不同），因此输入字典并非"每层 2 个 key"，而是按层类型变化：

      - compress_ratio > 1（C4A / C128A，主压缩注意力层）：
          主 KV   `model.layers.{i}.self_attn`              → MLAAttentionSpec
          SWA     `model.layers.{i}.self_attn.swa_cache`    → SlidingWindowMLASpec(block_size=64)
          state   `model.layers.{i}.self_attn.compressor.state_cache`
                                                            → SlidingWindowMLASpec(block_size=4 或 8)
          （仅 C4A，compress_ratio==4 才有 indexer）
          indexer `model.layers.{i}.self_attn.indexer.k_cache` → MLAAttentionSpec(compress_ratio=1)
      - compress_ratio <= 1（SWA-only 层）：主 KV 的 get_kv_cache_spec() 返回 None，
          只贡献 1 个 key：swa_cache（SlidingWindowMLASpec）。

    输入示例（一个 C4A 层 + 一个 SWA-only 层；C4A 层共 5 个 key）：
      kv_cache_spec = {
          "model.layers.0.self_attn":            MLAAttentionSpec(block_size=BS,
                                                  compress_ratio=4, model_version="deepseek_v4", ...),
          "model.layers.0.self_attn.swa_cache":  SlidingWindowMLASpec(block_size=64,
                                                  sliding_window=WS, ...),
          "model.layers.0.self_attn.compressor.state_cache":
                                                SlidingWindowMLASpec(block_size=4,
                                                  sliding_window=8, ...),
          "model.layers.0.self_attn.indexer.k_cache":
                                                MLAAttentionSpec(block_size=BS,
                                                  compress_ratio=1, ...),
          # ⚠️ 第 5 类：indexer 内部也持有自己的 CompressorStateCache，仅 C4A 层存在。
          # 其 (block_size=4, sliding_window=8) 与主 compressor state 完全相同，故在
          # 下方按 (block_size, sliding_window) 分桶时被归入同一组（输出示例 ③）。
          "model.layers.0.self_attn.indexer.compressor.state_cache":
                                                SlidingWindowMLASpec(block_size=4,
                                                  sliding_window=8, ...),
          "model.layers.1.self_attn.swa_cache": SlidingWindowMLASpec(block_size=64,
                                                  sliding_window=WS, ...),
          ...
      }

    输出示例（真实返回 4 个 UniformTypeKVCacheSpecs，顺序固定）：
      [
          UniformTypeKVCacheSpecs(kv_cache_specs={  # ① 全 MLA 组：主压缩 KV + indexer KV
              "model.layers.0.self_attn":            MLAAttentionSpec(...),
              "model.layers.0.self_attn.indexer.k_cache": MLAAttentionSpec(...),
              ...
          }),
          UniformTypeKVCacheSpecs(kv_cache_specs={  # ② SWA 组 (block_size=64, 同窗口)
              "model.layers.0.self_attn.swa_cache":  SlidingWindowMLASpec(...),
              "model.layers.1.self_attn.swa_cache":  SlidingWindowMLASpec(...),
              ...
          }),
          UniformTypeKVCacheSpecs(kv_cache_specs={  # ③ C4 压缩状态组 (block_size=4, sliding_window=8)
              # 注意：同一 C4A 层里有两份 CompressorStateCache，且 (block_size=4, sliding_window=8) 完全相同，故被分桶到同一组（head_dim 不同但允许）：
              #   - 主 compressor 的（head_dim=512 路径）
              #   - indexer 内 compressor 的（head_dim=128 路径，key 含 .indexer.）
              "model.layers.0.self_attn.compressor.state_cache": SlidingWindowMLASpec(...),
              "model.layers.0.self_attn.indexer.compressor.state_cache": SlidingWindowMLASpec(...),
              ...
          }),
          UniformTypeKVCacheSpecs(kv_cache_specs={  # ④ C128 压缩状态组 (block_size=8)
              "model.layers.{j}.self_attn.compressor.state_cache": SlidingWindowMLASpec(...),
              ...
          }),
      ]
    注：②/③/④ 是否都存在取决于模型实际含哪些 compress_ratio；它们都由下方
    按 (block_size, sliding_window) 的 SWA 分桶逻辑自动产生。
    """
    # 仅 DeepSeekV4 会同时包含 SlidingWindowMLASpec + MLAAttentionSpec。
    # 其他模型直接返回 None，走 get_kv_cache_groups 的下一条分支。
    if not any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    ):
        return None

    # 收集全部 MLA 全注意力层（主压缩 KV + indexer KV，两者都是 MLAAttentionSpec）。
    # 它们 token 数需求相同，被合并到同一个组（返回值第 ① 组）。
    mla_specs: dict[str, KVCacheSpec] = {}
    # 按 (block_size, sliding_window) 对 SWA MLA 层分组。
    # 不同 window size 或不同 block_size 的 SWA 层各自成组。
    # 例如 SWA 层 [swa_cache(64,WS)] 与 [C4-state(4,16)] 与 [C128-state(8,1024)]
    # 因 (block_size, sliding_window) 不同，被分到不同的组（返回值 ②/③/④）。
    grouped_swa_mla_specs: dict[tuple[int, int], dict[str, KVCacheSpec]] = defaultdict(
        dict
    )
    # NOTE: 这里按 (block_size, sliding_window) 对 SWA 层分组，能够把 swa_cache 层、
    # C4-state 层、C128-state 层分别分到不同组。仅用 block_size 和 sliding_window
    # 作为 key 比较脆弱，但目前够用。
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            # SWA 层按 (block_size, sliding_window) 分桶。
            # 例如 (64, WS) → SWA 组, (4, 16) → C4-state 组, (8, 1024) → C128-state 组。
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            # 全 MLA 层（含 indexer KV）放入同一个 dict，最终合并为一个 UniformTypeKVCacheSpecs。
            mla_specs[name] = spec

    assert len(mla_specs) > 0
    # 所有 MLA 全注意力层统一为一个 UniformTypeKVCacheSpecs。
    mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
    assert mla_uniform_spec is not None

    # 每组 SWA MLA 层各自统一为一个 UniformTypeKVCacheSpecs。
    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        # 同一 (block_size, sliding_window) 的层合并为一个 UniformType。
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    # 返回值顺序固定：第一个是全 MLA 组，后面依次为各组 SWA MLA（按分桶顺序）。
    return [mla_uniform_spec, *swa_uniform_specs]


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    # ⚠️ DeepSeek V4强制MLA
    full_mla_spec = grouped_specs[0]
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: 对齐各 group 的 layer tuple 数量。
    # 一个 layer tuple = 每种 page_size 的层各取一个，例如：
    #   full MLA group: 11个 C4 层 + 10个 C128 层 → 11个 layer tuple
    #                   (每个 tuple = [C4I, C4A, C128]，C128 不足 11 的用 padding 补)
    #   SWA group:      21层全部相同 page_size → 21个 layer tuple
    # ═══════════════════════════════════════════════════════════════════════════
    num_layer_tuples_per_group: list[int] = [
        g_spec.get_num_layer_tuples() for g_spec in grouped_specs
    ]
    # 用近似 GCD 找一个统一的 num_layer_tuples，使得各 group 向上取整后的
    # 总 padding 最小。full MLA group 的 tuple 数作为下界（不能少于此值）。
    num_layer_tuples = _approximate_gcd(
        num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
    )
    # 各组 tuple 数向上对齐到 num_layer_tuples 的整数倍（不足的用 padding 补）。
    num_layer_tuples_per_group = [
        round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: 对每个 SWA group，先将各层的 page_size 对齐到全 MLA group 的
    # 对应 page_size，再将层按 num_layer_tuples 切分为多个子 group。
    # ═══════════════════════════════════════════════════════════════════════════
    swa_mla_specs = grouped_specs[1:]
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in swa_mla_specs
        for spec in group.kv_cache_specs.values()
    )

    # 全 MLA group 的所有不重复 page_size（如 [C4, C128]），
    # 后续 SWA 层的 page 会被填充到这些值之一。
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        # SWA page 不能超过 MLA page，否则 packed layout 交错时无法对齐。
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # ── Step 2a: 为每个 SWA page_size 找到最近的、不小于它的 MLA page_size ──
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # ── Step 2b: 将 SWA 各层的 page_size 填充到目标值，并按目标大小分组 ──
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                # 填充：page 实际字节不够，用 page_size_padded 补齐到 candidate。
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): 当前约束：同一 SWA group 内，每种 page_size 对应的层数
        # 必须相同。这样不需要在部分 layer tuple 中做 layer 级别的 padding。
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # ── Step 2c: 将层按 num_layer_tuples 切分为多个子 group ──
        # 例如 num_layers_per_size=30, num_layer_tuples=10 → num_tuple_groups=3
        # 把 30 层等分到 3 个 group，每个 group 含 10 个 layer tuple。
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        # zip: 将每种 page_size 的层一一配对，形成 layer tuple 列表。
        # layers_per_size = {1000: [L0,L1,L2], 2000: [L3,L4,L5]}
        # → layer_tuples = [(L0,L3), (L1,L4), (L2,L5)]  每个元组是一个 layer tuple
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            # 步长交错取 tuple：group 0 取 index 0,3,6..., group 1 取 index 1,4,7...
            # 这样每个子 group 的层分布均匀，便于后续 parallel dispatch。
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # 把嵌套的 tuple 展平为一维 layer name 列表。
            group_layer_names = [
                name for layer_tuple in group_layer_tuples for name in layer_tuple
            ]
            group_layer_specs = {
                name: sm_spec.kv_cache_specs[name] for name in group_layer_names
            }
            # 子 group 内部所有层同类型（SlidingWindowMLASpec），可合并为 UniformType。
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, *swa_mla_groups]


def _annotate_eagle_groups_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    # Detection uses the merged MLA spec's model_version.
    if not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        return
    # DeepseekV4's MTP attention layer is always the last layer, and we flag whichever
    # group contains it.
    # FIXME(yifan): avoid/generalize this hacky check.
    last_layer = next(reversed(kv_cache_spec))
    for group in kv_cache_groups:
        if last_layer in group.layer_names:
            group.is_eagle_group = True
            break


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    # 输入 kv_cache_spec 是每一层对应的KVCacheSpec {层名: 该层KVCacheSpec}的字典。
    # 老路径（扁平化）：把所有 SlidingWindowSpec 强制转成 FullAttentionSpec，
    # 使所有层共用同一张 block table（窗口限制由 attention metadata 兜底）。
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    # 无注意力模型（kv_cache_spec 为空字典）：返回空列表，交由 KVCacheManager 特殊处理。
    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    # ⚠️ 所有层 spec 完全相同（注意：带/不带 sliding window 的 FullAttentionSpec被视为同一类型）。
    # 绝大多数模型走这里：所有层放进 1 个 group。返回一个 [ KVCacheGroupSpec(一个KVCacheSpec) ]
    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)

    # ⚠️ 所以KVCacheSpec的父基类相同 且 block_size一致。
    # 所有层 attention 类型相同（如全是 Full，或全是窗口一致的 SWAs算Full，或者MLA也算Full），只是 hidden size 可能不同。
    #  仍合并成 1 个 group，但 group spec 保留逐层差异。 返回一个 [ KVCacheGroupSpec( 一个UniformTypeKVCacheSpecs ) ]
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    # ⚠️ DeepSeekV4 特例：所有层 token 数需求相同，但类型/窗口尺寸各异（MLA + 多种 SWA）。
    # ❗️❗️❗️❗️group_and_unify_kv_cache_specs很重要！❗️❗️❗️❗️
    #     输出示例（真实返回 4 个 UniformTypeKVCacheSpecs，顺序固定）：
    #       [
    #           UniformTypeKVCacheSpecs(kv_cache_specs={  # ① 全 MLA 组：主压缩 KV + indexer KV
    #               "model.layers.0.self_attn":            MLAAttentionSpec(...),
    #               "model.layers.0.self_attn.indexer.k_cache": MLAAttentionSpec(...),
    #               ...
    #           }),
    #           UniformTypeKVCacheSpecs(kv_cache_specs={  # ② SWA 组 (block_size=64, 同窗口)
    #               "model.layers.0.self_attn.swa_cache":  SlidingWindowMLASpec(...),
    #               "model.layers.1.self_attn.swa_cache":  SlidingWindowMLASpec(...),
    #               ...
    #           }),
    #           UniformTypeKVCacheSpecs(kv_cache_specs={  # ③ C4 压缩状态组 (block_size=4, sliding_window=8)
    #               # 注意：同一 C4A 层里有两份 CompressorStateCache，且 (block_size=4, sliding_window=8) 完全相同，故被分桶到同一组（head_dim 不同但允许）：
    #               #   - 主 compressor 的（head_dim=512 路径）
    #               #   - indexer 内 compressor 的（head_dim=128 路径，key 含 .indexer.）
    #               "model.layers.0.self_attn.compressor.state_cache": SlidingWindowMLASpec(...),
    #               "model.layers.0.self_attn.indexer.compressor.state_cache": SlidingWindowMLASpec(...),
    #               ...
    #           }),
    #           UniformTypeKVCacheSpecs(kv_cache_specs={  # ④ C128 压缩状态组 (block_size=8)
    #               "model.layers.{j}.self_attn.compressor.state_cache": SlidingWindowMLASpec(...),
    #               ...
    #           }),
    #       ]
    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):
        # DeepseekV4 case: All layers need the same number of token slots,
        # yet some layers are full attention while others are sliding window
        # attention in different sizes. Need to group layers into multiple
        # UniformTypeKVCacheSpecs.
        #  ⚠️ DeepSeekV4 特例：所有layer所需的token slots数完全一致
        # 最终 KVCacheGroupSpec 列表（共 1 + 3×2 = 7 个）：
        #   ├─ [0] full_mla_group           # ① 整体 = 所有 MLA 类层（C4主KV+C128主KV+indexer KV），
        #   │                               #   作为一个 UniformTypeKVCacheSpecs，含 11 个 tuple
        #   ├─ [1] swa_64_subgroup_0        # ② 的一部分（前 11 个 tuple 的 swa_cache 层）
        #   ├─ [2] swa_64_subgroup_1        # ② 的一部分（后 11 个 tuple 的 swa_cache 层，padding 补）
        #   ├─ [3] c4state_subgroup_0       # ③ 的一部分（前 11 个 tuple 的 C4-state 层）
        #   ├─ [4] c4state_subgroup_1       # ③ 的一部分（后 11 个 tuple 的 C4-state 层）
        #   ├─ [5] c128state_subgroup_0     # ④ 的一部分
        #   └─ [6] c128state_subgroup_1     # ④ 的一部分
        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)

        # 为 eagle speculative decoding 标记/对齐 group（DSv4 专用）。
        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)
        return kv_cache_groups

    # ===== ⚠️ 通用混合注意力分支（如 Full + SWA 类型不同，且非 DSv4）=====
    # HiddenStateCacheSpec 是投机解码（EAGLE 类 extract_hidden_states）里
    # "借 KV cache 机制缓存模型 hidden states"的特殊标记层：它不算注意力、没有 K/V 双份维度，维度被偷换成 (num_hidden_states, hidden_size)，
    # 且 page 语义与普通注意力层不同。因此把它先抽出来，避免干扰后续物理 page 大小统一与分组逻辑。
    # （注册表见 single_type_kv_cache_manager.py：它不参与分组，base_spec 仅是占位）
    # Pull HiddenStateCacheSpec layers out before the general multi-group
    # path so they don't affect page-size unification or grouping.
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    filtered_spec = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, HiddenStateCacheSpec)
    }

    # ⚠️ KVCacheManager 只能分配"单一大小"的 block，
    # ⚠️因此必须把所有层的物理 page字节数统一，值得一看~~
    # 若无法统一（如不能整除且 backend 不支持 padded page）会直接报错。
    # As KVCacheManager can only allocate memory of one size, we need to unify
    # the page size of the layers. For cases cannot be unified, this function
    # will raise an error.
    filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)

    # ⚠️ 按 attention 类型把层切成多个 group（每组独立 block_table，但 page 字节数相同）。
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)


    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            new_bs = max(common_page // per_token, 1)
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))

    return groups


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def get_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, float]:
    """
    Get the group-aware KV cache token capacity and max concurrency.
    """
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    return int(max_concurrency * max_model_len), max_concurrency


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    if not kv_cache_groups:
        return 0

    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # UniformTypeKVCacheSpecs special case (single group, per-layer specs)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs.
        # They must already be page_size aligned and share a common padded
        # layer-tuple layout. Even groups with fewer actual tuples still reserve
        # the global number of tuple slots in the shared tensor layout.
        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).get_num_layer_tuples()
            for group in kv_cache_groups
        )

        total_max_mem_usage_bytes = 0
        for group in kv_cache_groups:
            group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)
            g_max_mem_usage_pages = group_spec.max_memory_usage_pages(vllm_config)
            g_max_mem_usage_page_bytes = (
                num_layer_tuples * g_max_mem_usage_pages * layer_tuple_bytes
            )
            total_max_mem_usage_bytes += g_max_mem_usage_page_bytes
        return total_max_mem_usage_bytes

    # General case: group_size pools, each shared by one layer per group
    # Memory = group_size * page_size * blocks_for_max_len
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return

    # Find the max_model_len that fits across all workers.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    # ⚠️
    # 把所有 worker（不同 PP stage / 不同 TP rank）的 spec 字典合并成"整模型"的一张表。
    # 同一层的 spec 必须在各 worker 完全一致，否则断言失败（KV cache 分配需全局一致）。
    # 注意：即便后面走 DeepSeekV4 多 group 支线，这一步也是必须的——它先把分散在
    # 各 worker 的 layer_name→spec 聚合成全局视图，后续分组才看得见完整模型结构。
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Check if the KV cache specs are registered correctly.
    # This is to prevent that some layers are initialized with unregistered specs.
    # 防呆：确保所有 spec 都在注册表里登记过（例如 MLAAttentionSpec / SlidingWindowMLASpec）。
    # 没登记的层说明 __init__ 时忘记调用 register，会导致后续裸奔。
    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    # ⚠️ 这是 DeepSeekV4 支线的总入口！
    # get_kv_cache_groups 内部按 spec 差异分派：
    #   - 全部层 spec 相同 → 1 个 group（绝大多数模型）
    #   - 父类相同且 block_size 一致（仅 hidden 不同）→ 1 个 group(UniformTypeKVCacheSpecs)
    #   - ★ DeepSeekV4：类型/窗口各异但 token 数需求相同 →
    #       group_and_unify_kv_cache_specs() 把层切成多个 UniformTypeKVCacheSpecs
    #       （[ 多个MLA Spec, 多个SWA MLA Spec, 多个SWA MLA Spec ]），并 _annotate_eagle_groups_deepseek_v4
    #   返回的 global_kv_cache_groups 顺序固定，是后面"多 group 共享一张大 tensor"布局的前提。
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    # 把全局 group 投影到每个 worker 自己拥有的层（PP 场景：各 stage 只持部分层）。
    # _project_kv_cache_groups_to_worker 会按 worker_spec 过滤 layer_names，并对
    # UniformTypeKVCacheSpecs 重建只含本 worker 层的子集。
    # 对 DeepSeekV4 而言：DSv4 通常不开 PP（或 PP stage 内仍含完整 multi-group 结构），
    # 这一投影通常不裁剪任何 group，但保障了 PP 下的正确性。
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    # 手动覆盖 block 数时，把"可用显存"换算成 override 对应的字节数，使后续
    # auto-fit / 内存检查 / 配置构建三处都基于同一有效容量，避免口径不一致。
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block(vllm_config, groups)
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    if vllm_config.model_config.original_max_model_len == -1:
        # max_model_len=-1 时自动二分搜索能塞进显存的最大上下文长度。
        # 用 per-worker 投影 group（考虑 PP 分片后每 stage 的真实 group 集合）做预算。
        # DSv4 多 group 情况下：_max_memory_usage_bytes_from_groups 走
        # "all groups are UniformTypeKVCacheSpecs" 特例（kv_cache_utils.py:2037），
        # 按 layer_tuple 共享布局 + 全局最大 layer_tuple 数算字节，正确反映 packed 浪费。
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    # 逐 worker 校验显存是否够装下 max_model_len 所需的 KV cache。
    # DSv4 多 group 的字节上限同样走上面 2037 的特例分支。
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    # ⚠️ 重要！
    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        # 每个 worker 用自己的投影 group + 自己的可用显存，生成一份 KVCacheConfig。
        # 对 DeepSeekV4：get_kv_cache_config_from_groups 内部会调到
        # _get_kv_cache_config_packed（别名 _get_kv_cache_config_deepseek_v4，kv_cache_utils.py:1443），
        # 即"一块大 tensor + 每 block 内各层 page 紧挨"的 packed 布局。
        # 多个 group（MLA/SWA/state）共享同一张 block table、同一块大 tensor，
        # 通过 offset + block_stride 寻址——这正是 DSv4 多 cache 能统一管理的关键。
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    # 木桶效应：所有 worker 必须用相同的 num_blocks（block table 要一致才能跨 rank 对齐）。
    # 取全局最小 block 数，并按比例收缩每个 tensor 的物理大小，避免给显存多的 worker 分配用不到的内存。
    # DSv4 多 group 下：kv_cache_tensors 是 packed 布局里那一块大 tensor（可能多个 KVCacheTensor 共享），
    # 收缩时按 num_blocks_old→min_num_blocks 线性缩放 size，保持 page 偏移关系不变。
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            max_model_len = vllm_config.model_config.max_model_len
            # GPU KV cache size in tokens = max_concurrency * max_model_len:
            # the total tokens of context the pool can hold at peak
            # utilization. Sourcing this from the concurrency calculation
            # handles hybrid layouts correctly.
            # DSv4 多 group 混合布局下，get_kv_cache_capacity 用 group-aware 并发度计算，
            # 正确反映"packed 一块大 tensor 撑起的并发上限"。
            num_tokens, max_concurrency = get_kv_cache_capacity(
                vllm_config, kv_cache_config
            )

            logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
            logger.info_once(
                "Maximum concurrency for %s tokens per request: %.2fx",
                f"{max_model_len:,}",
                max_concurrency,
            )

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency. Each `hash_block_size` hash is
    already chained over its entire prefix, so the hash at the last
    `hash_block_size` boundary of a `target_block_size` block uniquely
    fingerprints that block's prefix; we use it directly.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    the second 16-size hash already covers tokens 0-31, so it is the 32-size
    hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | B    | D     |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        # The last hash_block_size hash within the target block already chains
        # over the whole prefix, so it is the target block's hash.
        # ⚠️ 返回target_block内的最后一个hash block的hash值
        return self.block_hashes[(idx + 1) * self.scale_factor - 1]


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
