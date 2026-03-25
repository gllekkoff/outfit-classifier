import os
import base64
import csv
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

CLASSES = [
    "jacket", "coat", "blazer", "hoodie", "sweater",
    "t-shirt", "shirt", "top",
    "jeans", "trousers", "shorts", "skirt",
    "dress", "jumpsuit",
    "sneakers", "boots", "heels", "sandals", "shoes",
    "bag", "backpack", "hat", "cap", "scarf", "belt"
]

PROMPT = f"""
Identify all clothing items worn by the person.

STRICT RULES:
- Use ONLY these labels: {CLASSES}
- Do NOT use any words outside this list
- Do NOT invent or modify labels
- Only include items that are clearly visible
- If unsure, DO NOT include the item
- Do NOT infer hidden or partially visible clothing
- Do NOT describe colors, materials, or styles
- Do NOT repeat items
- Do NOT include duplicates
- Output must be a JSON array of strings only
- No explanations, no comments, no text before or after
- If no clothing is clearly visible, return []

FORMAT:
["item1", "item2"]

EXAMPLE:
["t-shirt", "jeans", "sneakers"]
"""

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def label_image(image_path, retries=3):
    img_b64 = encode_image(image_path)

    for attempt in range(retries):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    ],
                }],
            )

            return json.loads(response.output_text.strip())

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(2)

    return []

def clean_labels(labels):
    return list(set([l for l in labels if l in CLASSES]))

def to_multihot(labels):
    return [1 if c in labels else 0 for c in CLASSES]

def process_folder(folder, output_csv, limit=None):
    rows = []
    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]

    if limit:
        images = images[:limit]

    total = len(images)
    processed = 0

    print(f"Total images to process: {total}")

    for img_name in images:
        path = os.path.join(folder, img_name)
        processed += 1

        print(f"[{processed}/{total}] {img_name}")

        labels = label_image(path)
        labels = clean_labels(labels)
        vector = to_multihot(labels)

        rows.append([path] + vector)

        if processed % 20 == 0:
            print(f"Processed {processed}/{total}")

        time.sleep(0.7)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"] + CLASSES)
        writer.writerows(rows)

    print(f"Finished: {processed} images saved to {output_csv}")


# RUN with limits
process_folder("test", "test_labels.csv", limit=1000)