"""Download source data into ./data. Replace stubs with real fetchers."""
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def download_all() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    # TODO: implement per-source downloads for {{KG_NAME}}
    print(f"[download] wrote sources into {DATA_DIR}")

if __name__ == "__main__":
    download_all()
