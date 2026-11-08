from pathlib import Path

import httpx
from tqdm import tqdm
# www.repository.cam.ac.uk/items/6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8
API = "https://api.repository.cam.ac.uk/server/api/core"
ITEM = "6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8"
DEST = Path(__file__).parent.parent / "dataset"


def list_files(client):
    bundles = client.get(f"{API}/items/{ITEM}/bundles").json()["_embedded"]["bundles"]
    original = next(b for b in bundles if b["name"] == "ORIGINAL")
    url = original["_links"]["bitstreams"]["href"]
    return client.get(url, params={"size": 1000}).json()["_embedded"]["bitstreams"]


def download(client, file, bar):
    path = DEST / file["name"]
    if path.exists() and path.stat().st_size == file["sizeBytes"]:
        bar.update(file["sizeBytes"])
        return
    bar.set_postfix_str(file["name"])
    with client.stream("GET", file["_links"]["content"]["href"]) as r, path.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            f.write(chunk)
            bar.update(len(chunk))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        files = list_files(client)
        with tqdm(total=sum(f["sizeBytes"] for f in files), unit="B", unit_scale=True) as bar:
            for file in files:
                download(client, file, bar)


if __name__ == "__main__":
    main()
