import requests, os

URL = "https://zenodo.org/api/records/12646943/files/test.zip/content"
DEST = "model/data/raw/test.zip"
CHUNK_SIZE = 8 * 1024 * 1024
EXPECTED_SIZE = 6138851343

def download():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    headers = {}
    existing_size = 0
    if os.path.exists(DEST):
        existing_size = os.path.getsize(DEST)
        if existing_size >= EXPECTED_SIZE:
            print("[OK] test.zip already complete ({:.1f} GB)".format(existing_size/1e9))
            return
        headers["Range"] = "bytes={}-".format(existing_size)
        print("[RESUME] Existing: {:.1f} GB, resuming...".format(existing_size/1e9))
    else:
        print("[DOWNLOAD] ORCHID test.zip ({:.1f} GB)".format(EXPECTED_SIZE/1e9))
    
    r = requests.get(URL, stream=True, headers=headers, timeout=60)
    r.raise_for_status()
    
    with open(DEST, "ab" if existing_size > 0 else "wb") as f:
        downloaded = existing_size
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / EXPECTED_SIZE * 100
                print("  Progress: {:.1f}% ({:.1f}/{:.1f} GB)".format(
                    pct, downloaded/1e9, EXPECTED_SIZE/1e9))
    
    final_size = os.path.getsize(DEST)
    print("[OK] test.zip complete: {:.1f} GB".format(final_size/1e9))

if __name__ == "__main__":
    download()
