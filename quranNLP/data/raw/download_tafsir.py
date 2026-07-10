import os
import time
import requests

# Set paths
BASE_DIR = "quranNLP" if os.path.exists("quranNLP") else "."
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_all_tafsir():
    TAFSIR_ID = 169  # Ibn Kathir (English)
    all_tafsir_data = {"tafsirs": []}
    
    print("🚀 Initializing complete bulk fetch of Tafsir Ibn Kathir via Quran.com API...")
    
    # Loop through all 114 Surahs
    for surah in range(1, 115):
        url = f"https://api.quran.com/api/v4/tafsirs/{TAFSIR_ID}/by_chapter/{surah}"
        headers = {"Accept": "application/json", "User-Agent": "QuranNLP-Research-Agent"}
        
        print(f" -> Downloading Surah {surah}/114...")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            # Append each verse's tafsir text block
            for item in data.get("tafsirs", []):
                all_tafsir_data["tafsirs"].append({
                    "verse_key": item.get("verse_key"),
                    "text": item.get("text")
                })
        else:
            print(f"❌ Error on Surah {surah}: Status {response.status_code}")
            
        # Add a microscopic sleep step to be courteous to the API
        time.sleep(0.1)

    # Save complete array payload 
    output_path = os.path.join(RAW_DIR, "tafsir_ibn_kathir.json")
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_tafsir_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Complete database saved successfully to: {output_path}")

if __name__ == "__main__":
    fetch_all_tafsir()