import os

path = r'C:\Data\Code\nlp\vllm\.codebuddy\skills\source-analyzer\SKILL.md'

# Read with GBK encoding, write back as UTF-8
with open(path, 'r', encoding='gbk') as f:
    content = f.read()

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    verified = f.read()

print("Fixed! First 100 chars:")
print(verified[:150])
print()
print("Encoding is now UTF-8:", True)
