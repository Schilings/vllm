import os

def check_encoding(path):
    with open(path, 'rb') as f:
        raw = f.read()
    
    has_bom = raw[:3] == b'\xef\xbb\xbf'
    
    try:
        utf8_text = raw.decode('utf-8')
        utf8_ok = True
    except:
        utf8_text = 'FAILED'
        utf8_ok = False
    
    # Find description line
    desc_start = raw.find(b'description:')
    desc_end = raw.find(b'\n', desc_start) if desc_start >= 0 else -1
    desc_raw = raw[desc_start:desc_end] if desc_start >= 0 else b'N/A'
    
    return {
        'path': path,
        'has_bom': has_bom,
        'utf8_ok': utf8_ok,
        'desc_raw': desc_raw[:200].hex(),
        'desc_utf8': desc_raw.decode('utf-8', errors='replace')[:100] if utf8_ok else 'N/A'
    }

skills_dir = r'C:\Data\Code\nlp\vllm\.codebuddy\skills'
for skill in ['source-analyzer', 'code-commenter']:
    md = os.path.join(skills_dir, skill, 'SKILL.md')
    if os.path.exists(md):
        info = check_encoding(md)
        print(f'=== {skill} ===')
        print(f'Has BOM: {info["has_bom"]}')
        print(f'UTF-8 OK: {info["utf8_ok"]}')
        print(f'Description raw hex: {info["desc_raw"]}')
        print(f'Description (UTF-8): {info["desc_utf8"]}')
        print()
