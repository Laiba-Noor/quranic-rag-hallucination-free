import os
import zipfile
import requests

# Define our destination folder
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# URL Dictionary for target datasets
DATASETS = {
    "tanzil_uthmani": "https://tanzil.net/pub/download/quran-uthmani.xml",
    "qac_morphology": "https://corpus.quran.com/download/quranic-corpus-morphology-0.4.txt",
    "qrcd_dataset": "https://raw.githubusercontent.com/RanaMalhas/QRCD/main/data/qrcd_v1.1_train.json"
}

def download_file(url, destination):
    print(f"Downloading {url}...")
    # Add a User-Agent header to mimic a normal browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {destination}\n")
    else:
        print(f" Failed to download {url}. Status code: {response.status_code}")

# 1. Download Tanzil Text (XML format)
download_file(DATASETS["tanzil_uthmani"], os.path.join(RAW_DIR, "quran-uthmani.xml"))

# 2. Download Quranic Arabic Corpus Morphology Data
download_file(DATASETS["qac_morphology"], os.path.join(RAW_DIR, "quranic-corpus-morphology-0.4.txt"))

# 3. Download QRCD (Quranic Reading Comprehension Dataset)
download_file(DATASETS["qrcd_dataset"], os.path.join(RAW_DIR, "qrcd_v1.1_train.json"))

print(" A1 Core downloads complete! Check your 'data/raw' directory.")