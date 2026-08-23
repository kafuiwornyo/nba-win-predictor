# grabs the raw dataset. no api key needed, its just a csv on github
# https://github.com/NocturneBear/NBA-Data-2010-2024

import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_CSV_PATH, RAW_CSV_URL, DATA_DIR


def fetch_raw_data(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_CSV_PATH) and not force:
        print("already downloaded, skipping:", RAW_CSV_PATH)
        return RAW_CSV_PATH

    print("downloading...")
    urllib.request.urlretrieve(RAW_CSV_URL, RAW_CSV_PATH)
    mb = os.path.getsize(RAW_CSV_PATH) / (1024 * 1024)
    print(f"done, {mb:.1f} MB saved to {RAW_CSV_PATH}")
    return RAW_CSV_PATH


if __name__ == "__main__":
    fetch_raw_data()
