"""
Robust ORCHID train.zip downloader from Zenodo.
Uses the API content URL with resume support.
"""
import requests, os, sys

URL = "https://zenodo.org/api/records/12636426/files/train.zip/content"
DEST = "model/data/raw/train_new.zip"
CHUNK_SIZE = 8 * 1024 * 1024
EXPECTED_SIZE = 39745604871

def download():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    
    headers = {}
    existing_size = 0
    if os.path.exists(DEST):
        existing_size = os.path.getsize(DEST)
        if existing_size >= EXPECTED_SIZE:
            print("[OK] train_new.zip already complete ({:.1f} GB)".format(existing_size/1e9))
            return
        headers["Range"] = "bytes={}-".format(existing_size)
        print("[RESUME] Existing: {:.1f} GB, resuming...".format(existing_size/1e9))
    else:
        print("[DOWNLOAD] ORCHID train.zip ({:.1f} GB)".format(EXPECTED_SIZE/1e9))
    
    r = requests.get(URL, stream=True, headers=headers, timeout=60)
    r.raise_for_status()
    
    total = int(r.headers.get("content-length", EXPECTED_SIZE - existing_size))
    mode = "ab" if existing_size > 0 else "wb"
    
    with open(DEST, mode) as f:
        downloaded = existing_size
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / EXPECTED_SIZE * 100
                print("  Progress: {:.1f}% ({:.1f}/{:.1f} GB)".format(
                    pct, downloaded/1e9, EXPECTED_SIZE/1e9))
    
    final_size = os.path.getsize(DEST)
    if final_size < EXPECTED_SIZE * 0.99:
        print("[WARN] Incomplete: {:.1f} GB vs expected {:.1f} GB".format(
            final_size/1e9, EXPECTED_SIZE/1e9))
    else:
        print("[OK] Complete: {:.1f} GB".format(final_size/1e9))

if __name__ == "__main__":
    download()
