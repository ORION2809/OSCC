"""
Download ORCHID dataset from Zenodo using Python requests with proper headers.
"""

import requests
from pathlib import Path

DATA_ROOT = Path("model/data/raw")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/zip,application/octet-stream,*/*",
}

session = requests.Session()
session.headers.update(headers)

# Zenodo records
records = {
    "orchid_train": ("12636426", "train.zip"),   # 39.7 GB
    "orchid_val":   ("12646943", "val.zip"),     # 11.3 GB
    "orchid_test":  ("12646943", "test.zip"),    # 6.1 GB
}

for name, (record_id, filename) in records.items():
    dest = DATA_ROOT / filename
    if dest.exists():
        print(f"[SKIP] {filename} already exists ({dest.stat().st_size / 1024**3:.1f} GB)")
        continue
    
    url = f"https://zenodo.org/records/{record_id}/files/{filename}"
    print(f"\n[DOWNLOAD] {name}")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")
    
    try:
        r = session.get(url, stream=True, timeout=30, allow_redirects=True)
        r.raise_for_status()
        
        content_type = r.headers.get('Content-Type', '')
        content_length = r.headers.get('Content-Length')
        
        print(f"  Content-Type: {content_type}")
        print(f"  Content-Length: {content_length}")
        
        if 'text/html' in content_type:
            print(f"  [ERROR] Got HTML page instead of file. Zenodo may require manual download.")
            # Save a snippet for debugging
            with open(dest.with_suffix('.html'), 'w') as f:
                f.write(r.text[:5000])
            continue
        
        total = int(content_length) if content_length else None
        downloaded = 0
        
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded/1024**3:.1f}/{total/1024**3:.1f} GB)", end='')
        
        print(f"\n  [OK] Saved: {dest} ({downloaded / 1024**3:.1f} GB)")
        
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n\nDone.")
