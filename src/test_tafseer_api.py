import requests

def fetch_sample_tafsir():
    # 1. Ibn Kathir Tafsir ID on Quran.com API is usually 169
    # 2. Al-Jalalayn Tafsir ID is usually 16
    TAFSIR_ID = 169  
    VERSE_KEY = "1:1"  # Let's test Surah Al-Fatiha, Ayah 1
    
    url = f"https://api.quran.com/api/v4/tafsirs/{TAFSIR_ID}/by_ayah/{VERSE_KEY}"
    
    headers = {
        "Accept": "application/json",
        "User-Agent": "QuranNLP-Research-Agent"
    }
    
    print(f"Connecting to Tafsir API for Verse {VERSE_KEY}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        tafsir_text = data.get("tafsir", {}).get("text", "")
        print("\n API Connection Successful!")
        print(f"\nSample Tafsir Text Snippet:\n{tafsir_text[:300]}...")
    else:
        print(f" Connection failed. Status code: {response.status_code}")

if __name__ == "__main__":
    fetch_sample_tafsir()