import os
import re
import unicodedata
import xml.etree.ElementTree as ET

def clean_classical_arabic(text):
    """
    Standardizes orthographic forms and strips diacritics for Classical Arabic data mining.
    """
    if not text:
        return ""
        
    # 1. Strip common boilerplate HTML tags often found in API dumps
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Apply Unicode canonical decomposition (separates letters from diacritics)
    text = unicodedata.normalize('NFD', text)
    
    # 3. Explicitly strip Arabic diacritics (Tashkeel / Harakat)
    # This range handles Fathah, Dammah, Kasrah, Sukun, Shaddah, and Quranic ornamentation glyphs
    tashkeel_pattern = re.compile(r'[\u064B-\u0652\u0653-\u065F\u0670]')
    text = re.sub(tashkeel_pattern, '', text)
    
    # 4. Standardize variations of Alif (أ, إ, آ, ٱ) into a bare Alif (ا)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    
    # 5. Standardize Alef Maksura (ى) into Yeh (ي)
    text = re.sub(r'ى', 'ي', text)
    
    # 6. Bring back to Unicode Canonical Composition form
    text = unicodedata.normalize('NFC', text)
    
    # Clear out redundant extra spaces
    return " ".join(text.split())

def parse_tanzil_xml(xml_path):
    """
    Safely reads your downloaded Tanzil XML file and runs the text pipeline.
    """
    print(f"Processing Tanzil XML text from: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    cleaned_corpus = {}
    
    # Navigate standard Tanzil XML hierarchy (surah -> ayah)
    for surah in root.findall('.//surah'):
        surah_num = surah.get('index')
        for ayah in surah.findall('.//ayah'):
            ayah_num = ayah.get('index')
            verse_key = f"{surah_num}:{ayah_num}"
            raw_text = ayah.get('text')
            
            cleaned_text = clean_classical_arabic(raw_text)
            cleaned_corpus[verse_key] = cleaned_text
            
    return cleaned_corpus

if __name__ == "__main__":
    # Target directory routing 
    xml_file_path = "quranNLP/data/raw/quran-uthmani-min.xml"
    if not os.path.exists(xml_file_path):
        xml_file_path = "data/raw/quran-uthmani-min.xml"
        
    try:
        processed_data = parse_tanzil_xml(xml_file_path)
        print("\n✅ Pipeline executed successfully across Tanzil corpus!")
        print(f"Total Verses Processed: {len(processed_data)}")
        print(f"Sample Cleaned Verse (1:1): {processed_data.get('1:1')}")
    except Exception as e:
        print(f"❌ Error encountered: {e}")