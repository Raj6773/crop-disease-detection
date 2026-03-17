import json
from deep_translator import GoogleTranslator
from pathlib import Path

# Input and Output
input_file = Path("diseases.json")
output_file = Path("diseases_multilang.json")

# Load original diseases.json
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def safe_translate(text, lang):
    if not text:
        return ""
    try:
        return GoogleTranslator(source="en", target=lang).translate(text)
    except Exception as e:
        print(f"Translation failed for '{text}' -> {lang}: {e}")
        return text

new_data = {}

print("Starting translation...")

for i, (key, val) in enumerate(data.items(), start=1):
    entry = dict(val)
    print(f"{i}/{len(data)} Translating {key}...")
    for field in ["display_name", "cause", "treatment", "prevention", "youtube_query"]:
        text_val = entry.get(field, "")
        entry[field] = {
            "en": text_val,
            "hi": safe_translate(text_val, "hi"),
            "te": safe_translate(text_val, "te")
        }
    new_data[key] = entry

# Save output
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Multilingual file saved as {output_file}")
