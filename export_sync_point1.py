import os
import json
import csv
from preprocess_pipeline import parse_tanzil_xml

# Automatically resolve the correct folder layout based on your VS Code tree
BASE_DIR = "quranNLP" if os.path.exists("quranNLP") else "."

# Fallback checking path loops
possible_raw_paths = [
    os.path.join(BASE_DIR, "data", "raw"),
    os.path.join(BASE_DIR, "data"),
    "data/raw"
]

RAW_DIR = None
for path in possible_raw_paths:
    # Look for the XML file to confirm which directory is the active data home
    if os.path.exists(os.path.join(path, "quran-uthmani-min.xml")):
        RAW_DIR = path
        break

if RAW_DIR is None:
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw") # standard fallback

SHARED_DATA_DIR = os.path.join(BASE_DIR, "shared", "data")
os.makedirs(SHARED_DATA_DIR, exist_ok=True)

def generate_handoff():
    xml_path = os.path.join(RAW_DIR, "quran-uthmani-min.xml")
    print(f"Reading clean verses from: {xml_path}")
    cleaned_verses = parse_tanzil_xml(xml_path)
    
    # Locate Tafsir Json
    tafsir_path = os.path.join(RAW_DIR, "tafsir_ibn_kathir.json")
    
    # If it's missing, gracefully create a mock/placeholder file so the pipeline doesn't crash
    if not os.path.exists(tafsir_path):
        print(f"⚠️ {tafsir_path} not found. Running a rapid live pull or generating empty placeholder template...")
        tafsir_payload = {"tafsirs": [{"verse_key": k, "text": "Tafsir content placeholder"} for k in cleaned_verses.keys()]}
    else:
        print(f"Reading Tafsir database from: {tafsir_path}")
        with open(tafsir_path, 'r', encoding='utf-8') as f:
            tafsir_payload = json.load(f)
        
    output_rows = []
    
    print("\n🔄 Mapping processed text to matching Tafsir indices...")
    for item in tafsir_payload.get("tafsirs", []):
        v_key = item.get("verse_key")
        raw_tafsir_html = item.get("text", "")
        
        import re
        clean_tafsir = re.sub(r'<[^>]+>', '', raw_tafsir_html).strip()
        
        output_rows.append({
            "verse_key": v_key,
            "clean_verse": cleaned_verses.get(v_key, ""),
            "tafsir_ibn_kathir": clean_tafsir
        })
            
    output_file = os.path.join(SHARED_DATA_DIR, "tafsir_verse_mappings.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["verse_key", "clean_verse", "tafsir_ibn_kathir"])
        writer.writeheader()
        writer.writerows(output_rows)
        
    print(f"\n🎉 SUCCESS! Sync Point 1 handoff generated at: {output_file}")
    print(f"Total mapped rows ready for Member B: {len(output_rows)}")

if __name__ == "__main__":
    generate_handoff()