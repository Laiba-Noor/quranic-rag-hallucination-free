import os
import xml.etree.ElementTree as ET
import csv

BASE_DIR = "quranNLP" if os.path.exists("quranNLP") else "."
XML_PATH = os.path.join(BASE_DIR, "data", "raw", "quran-uthmani-min.xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "shared", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "tafsir_verse_mappings.csv")

def clean_arabic_text(text):
    if not text:
        return ""
    # Remove diacritics (Tashkeel)
    diacritics = re.compile(r'[\u064B-\u0652]')
    text = re.sub(diacritics, '', text)
    # Normalize Alifs
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    return text.strip()

def preprocess_tanzil_xml():
    if not os.path.exists(XML_PATH):
        print(f"ERROR: XML file not found at {XML_PATH}")
        return

    print(f"Processing Tanzil XML text from: {XML_PATH}")
    
    try:
        tree = ET.parse(XML_PATH)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR parsing XML file: {e}")
        return

    processed_verses = []
    
    # Iterate through sura and aya tags
    for sura in root.findall('.//sura'):
        sura_index = sura.get('index')
        for aya in sura.findall('.//aya'):
            aya_index = aya.get('index')
            # Critical Fix: Read from the 'text' attribute instead of inner text
            raw_text = aya.get('text')
            
            if raw_text:
                cleaned_text = clean_arabic_text(raw_text)
                verse_key = f"{sura_index}:{aya_index}"
                
                processed_verses.append({
                    "verse_key": verse_key,
                    "clean_verse": cleaned_text,
                    "tafsir_ibn_kathir": f"Tafsir text placeholder for verse {verse_key}"
                })

    # Save to destination directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["verse_key", "clean_verse", "tafsir_ibn_kathir"])
        writer.writeheader()
        writer.writerows(processed_verses)

    print("\nPipeline executed successfully across Tanzil corpus!")
    print(f"Total Verses Processed: {len(processed_verses)}")
    if len(processed_verses) > 0:
        print(f"Sample Cleaned Verse (1:1): {processed_verses[0]['clean_verse']}")

if __name__ == "__main__":
    import re
    preprocess_tanzil_xml()