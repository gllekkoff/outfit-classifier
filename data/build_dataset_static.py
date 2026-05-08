import os
import re
import sys
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

STATIC_CLASSES = [
    "dress",
    "sleeveless_top",
    "t_shirt",
    "shirt",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "blazer",
    "cardigan",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
    "leggings",
    "sneakers",
    "boots",
    "heels",
    "sandals",
    "bag",
    "hat",
    "scarf",
    "belt",
    "sunglasses",
]

ALIASES = {
    "sleeveless_top": "sleeveless_top",
    "tank_top": "sleeveless_top",
    "tanktop": "sleeveless_top",
    "tank": "sleeveless_top",
    "halter_top": "sleeveless_top",
    "t-shirt": "t_shirt",
    "tshirt": "t_shirt",
    "tee": "t_shirt",
    "t_shirt": "t_shirt",
    "pants": "trousers",
    "trouser": "trousers",
    "slacks": "trousers",
    "sneaker": "sneakers",
    "sneakers": "sneakers",
    "trainer": "sneakers",
    "trainers": "sneakers",
    "boot": "boots",
    "boots": "boots",
    "ankle_boot": "boots",
    "ankle_boots": "boots",
    "heel": "heels",
    "heels": "heels",
    "high_heels": "heels",
    "pumps": "heels",
    "wedge_heels": "heels",
    "sandal": "sandals",
    "sandals": "sandals",
    "cap": "hat",
    "handbag": "bag",
    "purse": "bag",
    "backpack": "bag",
    "shoulder_bag": "bag",
    "clutch": "bag",
}


SYSTEM_PROMPT = f"""
You are a fashion item detector.

Given an image, detect ONLY these fixed classes:
{", ".join(STATIC_CLASSES)}

Class definitions:
- dress means a one-piece dress.
- sleeveless_top means a sleeveless upper-body garment such as a tank top, halter top, or sleeveless top.
- Do NOT use sleeveless_top for t-shirts, shirts, sweaters, hoodies, jackets, coats, blazers, or cardigans.
- t_shirt means a casual short-sleeve or long-sleeve t-shirt.
- shirt means a button-up shirt or blouse-like shirt.
- sweater means knitted or warm pullover.
- hoodie means hoodie with hood.
- jacket means short outerwear jacket.
- coat means longer outerwear coat.
- blazer means structured formal jacket.
- cardigan means open-front knitwear.
- trousers means formal or casual pants that are not jeans or leggings.
- jeans means denim jeans.
- shorts means short bottoms.
- skirt means skirt.
- leggings means tight stretch pants.
- sneakers means sport or casual sneakers only.
- boots means boots only.
- heels means high heels, pumps, or wedge heels.
- sandals means open sandals only.
- Do NOT use a generic shoes label. Choose sneakers, boots, heels, or sandals only if clearly visible.
- bag includes handbag, purse, clutch, shoulder bag, and backpack.
- hat includes hat and cap.
- scarf means scarf.
- belt means visible belt.
- sunglasses means sunglasses.

Return ONLY a valid JSON object with EXACTLY these keys:
{json.dumps({cls: 0 for cls in STATIC_CLASSES}, indent=2)}

Rules:
- Use 1 if the item is clearly visible.
- Use 0 if the item is not visible.
- Do NOT add new labels.
- Do NOT rename labels.
- Do NOT include explanations.
- Do NOT include markdown.
- Return pure JSON only.
"""


def encode_image(path: Path, max_side: int = 1024) -> tuple[str, str]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size

        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64, "image/jpeg"


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = label.replace("-", "_").replace(" ", "_").replace("/", "_")

    while "__" in label:
        label = label.replace("__", "_")

    return ALIASES.get(label, label)


def clean_model_labels(labels: dict) -> dict[str, int]:
    clean = {cls: 0 for cls in STATIC_CLASSES}

    if not isinstance(labels, dict):
        return clean

    for key, value in labels.items():
        normalized_key = normalize_label(str(key))

        if normalized_key not in clean:
            continue

        if isinstance(value, (int, float, bool)):
            clean[normalized_key] = max(clean[normalized_key], int(bool(value)))

    return clean


def is_dress_only(labels: dict) -> bool:
    clean_labels = clean_model_labels(labels)

    if clean_labels.get("dress", 0) != 1:
        return False

    for cls in STATIC_CLASSES:
        if cls == "dress":
            continue

        if clean_labels.get(cls, 0) == 1:
            return False

    return True


def call_vision(
    client: OpenAI,
    image_path: Path,
    max_retries: int = 4,
) -> dict[str, int] | None:
    b64, media_type = encode_image(image_path)

    raw = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=700,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}",
                                    "detail": "low",
                                },
                            },
                            {
                                "type": "text",
                                "text": "Detect the visible fashion items using only the fixed classes.",
                            },
                        ],
                    },
                ],
            )

            raw = response.choices[0].message.content.strip()

            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            labels = json.loads(raw)
            return clean_model_labels(labels)

        except json.JSONDecodeError as e:
            log.warning(f"[{image_path.name}] JSON parse error, attempt {attempt}: {e}")
            log.debug(f"Raw response: {raw!r}")

        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)

            if status == 429:
                wait = 2 ** attempt * 5
                log.warning(f"[{image_path.name}] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if status and status >= 500:
                wait = 2 ** attempt
                log.warning(f"[{image_path.name}] Server error {status}. Waiting {wait}s...")
                time.sleep(wait)
                continue

            log.error(f"[{image_path.name}] Unrecoverable error: {e}")
            return None

        time.sleep(2 ** attempt)

    log.error(f"[{image_path.name}] All {max_retries} retries exhausted. Skipping.")
    return None


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)

        rows = data.get("rows", {})
        saved_classes = data.get("classes", [])

        if saved_classes and saved_classes != STATIC_CLASSES:
            log.warning(
                "Checkpoint classes are different from current STATIC_CLASSES. "
                "If you changed labels, use a new checkpoint and relabel images."
            )

        log.info(f"Resuming from checkpoint: {len(rows):,} images already done.")

        return {
            "rows": rows,
            "classes": STATIC_CLASSES,
        }

    return {
        "rows": {},
        "classes": STATIC_CLASSES,
    }


def save_checkpoint(path: Path, rows: dict):
    tmp = path.with_suffix(".tmp")

    with open(tmp, "w") as f:
        json.dump(
            {
                "rows": rows,
                "classes": STATIC_CLASSES,
            },
            f,
            indent=2,
        )

    tmp.replace(path)


def build_csv_from_rows(rows: dict, output_csv: Path, remove_dress_only: bool = True):
    csv_rows = []
    deleted_dress_only = 0

    for rel_path, labels in rows.items():
        clean_labels = clean_model_labels(labels)

        if remove_dress_only and is_dress_only(clean_labels):
            deleted_dress_only += 1
            continue

        row = {"image_path": rel_path}

        for cls in STATIC_CLASSES:
            row[cls] = clean_labels.get(cls, 0)

        csv_rows.append(row)

    df = pd.DataFrame(csv_rows, columns=["image_path"] + STATIC_CLASSES)

    if not df.empty:
        df[STATIC_CLASSES] = df[STATIC_CLASSES].fillna(0).astype(int)

    df.to_csv(output_csv, index=False)

    log.info(
        f"Saved {output_csv} "
        f"({len(df):,} rows x {len(STATIC_CLASSES)} label cols)"
    )

    print("\nDress-only filtering")
    print(f"Deleted dress-only rows from CSV: {deleted_dress_only}")
    print(f"Remaining rows in CSV: {len(df)}")

    return df, deleted_dress_only


def main():
    parser = argparse.ArgumentParser(
        description="Static fashion label dataset builder."
    )

    parser.add_argument(
        "--image_dir",
        required=True,
        help="Root folder of images",
    )

    parser.add_argument(
        "--output_csv",
        default="train_labels_static_v2.csv",
        help="Output CSV path",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoint_static_v2.json",
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save checkpoint every N images",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Parallel API calls. Use 1 for sequential.",
    )

    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key, or set OPENAI_API_KEY env var",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(SUPPORTED_EXT),
    )

    parser.add_argument(
        "--keep_dress_only",
        action="store_true",
        help="Keep rows where dress is the only positive label.",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)
    checkpoint_path = Path(args.checkpoint)

    if not image_dir.exists():
        sys.exit(f"ERROR: image_dir '{image_dir}' does not exist.")

    exts = {e.lower() for e in args.extensions}

    all_images = sorted([
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in exts
    ])

    log.info(f"Found {len(all_images):,} images in '{image_dir}'")

    checkpoint = load_checkpoint(checkpoint_path)
    done_rows = checkpoint["rows"]

    pending = [
        p for p in all_images
        if str(p.relative_to(image_dir)) not in done_rows
    ]

    log.info(f"Already processed : {len(done_rows):,}")
    log.info(f"Remaining         : {len(pending):,}")

    if pending:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

        if not api_key:
            sys.exit("ERROR: Set OPENAI_API_KEY environment variable or pass --api_key")

        client = OpenAI(api_key=api_key, max_retries=0)

        processed_since_checkpoint = 0

        def process_one(img_path: Path):
            rel = str(img_path.relative_to(image_dir))
            labels = call_vision(
                client=client,
                image_path=img_path,
                max_retries=args.max_retries,
            )
            return rel, labels

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(process_one, img_path): img_path
                for img_path in pending
            }

            with tqdm(total=len(pending), desc="Labelling images", unit="img") as pbar:
                for future in as_completed(futures):
                    rel, labels = future.result()

                    if labels is None:
                        log.warning(f"Skipped after failed retries: {rel}")
                        labels = {cls: 0 for cls in STATIC_CLASSES}

                    labels = clean_model_labels(labels)
                    done_rows[rel] = labels

                    processed_since_checkpoint += 1
                    pbar.update(1)

                    if processed_since_checkpoint % args.checkpoint_every == 0:
                        save_checkpoint(checkpoint_path, done_rows)
                        log.info(
                            f"Checkpoint saved "
                            f"({len(done_rows):,} total rows)"
                        )

        save_checkpoint(checkpoint_path, done_rows)
        log.info("Final checkpoint saved.")

    else:
        log.info("Nothing to do: all images already processed.")

    remove_dress_only = not args.keep_dress_only

    df, deleted_dress_only = build_csv_from_rows(
        rows=done_rows,
        output_csv=output_csv,
        remove_dress_only=remove_dress_only,
    )

    if not df.empty:
        counts = df[STATIC_CLASSES].sum().sort_values(ascending=False)

        print("\nLabel summary")
        print(counts[counts > 0].to_string())

    print("\nStatic classes:")
    print(STATIC_CLASSES)


if __name__ == "__main__":
    main()
