import base64
import io
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from tqdm.auto import tqdm


DEFAULT_CHECKPOINT = Path("checkpoints/best_model.pth")
DEFAULT_TEST_CSV = Path("results/test_split.csv")
DEFAULT_TRAIN_CSV = Path("results/train_split.csv")


class TimmMultiLabelModel(nn.Module):
    def __init__(self, timm_name, num_classes, pretrained=False, image_size=None):
        super().__init__()
        self.timm_name = timm_name
        create_kwargs = {"pretrained": pretrained, "num_classes": 0}
        if image_size is not None:
            create_kwargs["img_size"] = int(image_size)
        try:
            self.backbone = timm.create_model(timm_name, **create_kwargs)
        except TypeError:
            create_kwargs.pop("img_size", None)
            self.backbone = timm.create_model(timm_name, **create_kwargs)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)
        return self.head(features)


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path=DEFAULT_CHECKPOINT, device=None):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = ["timm_name", "class_names", "model_state_dict", "image_size"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing required keys: {missing}")

    model = TimmMultiLabelModel(
        checkpoint["timm_name"],
        len(checkpoint["class_names"]),
        image_size=checkpoint.get("image_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint, device


def checkpoint_transform(checkpoint):
    size = int(checkpoint["image_size"])
    mean = tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406)))
    std = tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225)))
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]), size


def prepare_image(image, size):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=(255, 255, 255))


def predict_image(model, checkpoint, image, device=None, threshold=None, use_tta=True):
    device = device or next(model.parameters()).device
    transform, size = checkpoint_transform(checkpoint)
    default_threshold = float(checkpoint.get("threshold", 0.5) if threshold is None else threshold)
    per_class_thresholds = checkpoint.get("per_class_thresholds", None)

    prepared = prepare_image(image, size)
    tensor = transform(prepared).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        if use_tta:
            logits = (logits + model(TF.hflip(tensor))) / 2.0
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    latency_ms = (time.perf_counter() - start) * 1000

    class_names = list(checkpoint["class_names"])
    predictions = []
    for label, prob in zip(class_names, probs):
        t = float(per_class_thresholds[label]) if per_class_thresholds and label in per_class_thresholds else default_threshold
        predictions.append({
            "label": label,
            "probability": float(prob),
            "predicted": bool(float(prob) >= t),
            "threshold": t,
        })
    predictions.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "model_name": checkpoint.get("model_name", "model"),
        "threshold": default_threshold,
        "per_class_thresholds_used": per_class_thresholds is not None,
        "image_size": size,
        "latency_ms": latency_ms,
        "predictions": predictions,
    }


def load_split(path=DEFAULT_TEST_CSV):
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")
    df = pd.read_csv(split_path)
    if "filename" not in df.columns:
        raise ValueError(f"{split_path} must contain a filename column.")
    class_names = [c for c in df.columns if c != "filename"]
    return df, class_names


def validate_classes(checkpoint, class_names):
    checkpoint_classes = list(checkpoint["class_names"])
    if checkpoint_classes != list(class_names):
        raise ValueError(
            "Class mismatch between checkpoint and CSV. "
            f"checkpoint={len(checkpoint_classes)} classes, csv={len(class_names)} classes"
        )


def threshold_vector(threshold, class_names):
    if isinstance(threshold, dict):
        return np.array([float(threshold[name]) for name in class_names], dtype=np.float32)
    values = np.asarray(threshold, dtype=np.float32)
    if values.ndim == 0:
        return float(values)
    if len(values) != len(class_names):
        raise ValueError(f"Expected {len(class_names)} thresholds, got {len(values)}")
    return values


def image_dir_for_checkpoint(checkpoint, dataset_dir="dataset"):
    return Path(dataset_dir) / f"train_{int(checkpoint['image_size'])}"


def predict_file(model, checkpoint, image_path, device=None, threshold=None, use_tta=True):
    with Image.open(image_path) as image:
        return predict_image(model, checkpoint, image, device=device, threshold=threshold, use_tta=use_tta)


def evaluate_checkpoint(checkpoint_path, split_csv=DEFAULT_TEST_CSV, dataset_dir="dataset", batch_size=4, device=None, use_tta=None):
    model, checkpoint, device = load_checkpoint(checkpoint_path, device=device)
    df, class_names = load_split(split_csv)
    validate_classes(checkpoint, class_names)
    image_dir = image_dir_for_checkpoint(checkpoint, dataset_dir)
    transform, size = checkpoint_transform(checkpoint)
    threshold = checkpoint.get("per_class_thresholds") or float(checkpoint.get("threshold", 0.5))
    use_tta = bool(checkpoint.get("use_tta_eval", False)) if use_tta is None else bool(use_tta)

    probs = []
    targets = []
    missing = []
    for start in tqdm(range(0, len(df), batch_size), desc=f"Evaluating {checkpoint.get('model_name', Path(checkpoint_path).stem)}"):
        rows = df.iloc[start:start + batch_size]
        images = []
        batch_targets = []
        for _, row in rows.iterrows():
            path = image_dir / row["filename"]
            if not path.is_file():
                missing.append(str(path))
                continue
            with Image.open(path) as image:
                images.append(transform(prepare_image(image, size)))
            batch_targets.append(row[class_names].values.astype(np.float32))
        if not images:
            continue
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            logits = model(batch)
            if use_tta:
                logits = (logits + model(TF.hflip(batch))) / 2.0
            probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(np.vstack(batch_targets).astype(int))

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images. First missing file: {missing[0]}")
    if not probs:
        raise ValueError("No images were evaluated.")

    probs = np.vstack(probs)
    targets = np.vstack(targets)
    preds = (probs >= threshold_vector(threshold, class_names)).astype(int)
    per_class = f1_score(targets, preds, average=None, zero_division=0)
    return {
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint.get("model_name", Path(checkpoint_path).stem),
        "image_size": int(checkpoint["image_size"]),
        "threshold": "per_class" if isinstance(threshold, dict) else float(threshold),
        "per_class_thresholds_used": isinstance(threshold, dict),
        "tta_eval": use_tta,
        "samples": int(len(targets)),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "samples_f1": f1_score(targets, preds, average="samples", zero_division=0),
        "precision_micro": precision_score(targets, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(targets, preds, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(targets, preds),
        "per_class_f1": dict(zip(class_names, per_class)),
        "probs": probs,
        "targets": targets,
        "preds": preds,
    }


def evaluate_pretrained_backbone_baseline(
    checkpoint_path=DEFAULT_CHECKPOINT,
    split_csv=DEFAULT_TEST_CSV,
    dataset_dir="dataset",
    batch_size=4,
    device=None,
    seed=42,
    limit=None,
):
    reference_model, checkpoint, device = load_checkpoint(checkpoint_path, device=device)
    del reference_model
    df, class_names = load_split(split_csv)
    if limit is not None:
        df = df.head(int(limit)).copy()
    validate_classes(checkpoint, class_names)

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = TimmMultiLabelModel(
        checkpoint["timm_name"],
        len(class_names),
        pretrained=True,
        image_size=checkpoint.get("image_size"),
    )
    model.to(device).eval()

    image_dir = image_dir_for_checkpoint(checkpoint, dataset_dir)
    transform, size = checkpoint_transform(checkpoint)
    threshold = float(checkpoint.get("threshold", 0.5))

    probs = []
    targets = []
    missing = []
    for start in tqdm(range(0, len(df), batch_size), desc="Evaluating pretrained backbone baseline"):
        rows = df.iloc[start:start + batch_size]
        images = []
        batch_targets = []
        for _, row in rows.iterrows():
            path = image_dir / row["filename"]
            if not path.is_file():
                missing.append(str(path))
                continue
            with Image.open(path) as image:
                images.append(transform(prepare_image(image, size)))
            batch_targets.append(row[class_names].values.astype(np.float32))
        if not images:
            continue
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(np.vstack(batch_targets).astype(int))

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images. First missing file: {missing[0]}")
    if not probs:
        raise ValueError("No images were evaluated.")

    probs = np.vstack(probs)
    targets = np.vstack(targets)
    preds = (probs >= threshold).astype(int)
    per_class = f1_score(targets, preds, average=None, zero_division=0)
    return {
        "model_name": "pretrained_convnext_random_outfit_head",
        "image_size": int(checkpoint["image_size"]),
        "threshold": threshold,
        "samples": int(len(targets)),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "samples_f1": f1_score(targets, preds, average="samples", zero_division=0),
        "precision_micro": precision_score(targets, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(targets, preds, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(targets, preds),
        "per_class_f1": dict(zip(class_names, per_class)),
        "probs": probs,
        "targets": targets,
        "preds": preds,
    }


def frequency_baseline(split_csv=DEFAULT_TEST_CSV, train_csv=DEFAULT_TRAIN_CSV, threshold=0.5):
    test_df, class_names = load_split(split_csv)
    train_path = Path(train_csv)
    if train_path.is_file():
        train_df, train_classes = load_split(train_path)
        if train_classes != class_names:
            raise ValueError("Train and test split class columns do not match.")
        prevalence = train_df[class_names].mean(axis=0).values.astype(np.float32)
    else:
        prevalence = test_df[class_names].mean(axis=0).values.astype(np.float32)

    targets = test_df[class_names].values.astype(int)
    preds = np.tile((prevalence >= threshold).astype(int), (len(targets), 1))
    return {
        "model_name": "label_frequency_baseline",
        "image_size": None,
        "threshold": threshold,
        "samples": int(len(targets)),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "samples_f1": f1_score(targets, preds, average="samples", zero_division=0),
        "precision_micro": precision_score(targets, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(targets, preds, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(targets, preds),
    }


def huggingface_zero_shot_baseline(
    split_csv=DEFAULT_TEST_CSV,
    dataset_dir="dataset",
    image_size=512,
    model_id="openai/clip-vit-base-patch32",
    batch_size=8,
    device=None,
    top_k=None,
    limit=None,
):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("Install transformers to run the Hugging Face zero-shot baseline.") from exc

    device = device or get_device()
    df, class_names = load_split(split_csv)
    if limit is not None:
        df = df.head(int(limit)).copy()

    labels_per_image = df[class_names].sum(axis=1).astype(int)
    if top_k is None:
        top_k = max(1, int(round(labels_per_image.mean())))

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def feature_tensor(output):
        if torch.is_tensor(output):
            return output
        for name in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, name, None)
            if torch.is_tensor(value):
                return value
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(f"Cannot extract tensor features from {type(output)}")

    prompts = [f"a photo of {label.replace('_', ' ')}" for label in class_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = feature_tensor(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_dir = Path(dataset_dir) / f"train_{int(image_size)}"
    probs = []
    targets = []
    missing = []
    for start in tqdm(range(0, len(df), batch_size), desc=f"Evaluating HF {model_id}"):
        rows = df.iloc[start:start + batch_size]
        images = []
        batch_targets = []
        for _, row in rows.iterrows():
            path = image_dir / row["filename"]
            if not path.is_file():
                missing.append(str(path))
                continue
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
            batch_targets.append(row[class_names].values.astype(np.float32))
        if not images:
            continue

        image_inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = feature_tensor(model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            batch_probs = similarities.softmax(dim=1).cpu().numpy()
        probs.append(batch_probs)
        targets.append(np.vstack(batch_targets).astype(int))

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images. First missing file: {missing[0]}")
    if not probs:
        raise ValueError("No images were evaluated.")

    probs = np.vstack(probs)
    targets = np.vstack(targets)
    preds = np.zeros_like(probs, dtype=int)
    top_indices = np.argsort(-probs, axis=1)[:, :top_k]
    for row_idx, cols in enumerate(top_indices):
        preds[row_idx, cols] = 1

    per_class = f1_score(targets, preds, average=None, zero_division=0)
    return {
        "model_name": f"huggingface_zero_shot_{model_id}",
        "image_size": int(image_size),
        "threshold": f"top_{top_k}",
        "samples": int(len(targets)),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "samples_f1": f1_score(targets, preds, average="samples", zero_division=0),
        "precision_micro": precision_score(targets, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(targets, preds, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(targets, preds),
        "per_class_f1": dict(zip(class_names, per_class)),
        "probs": probs,
        "targets": targets,
        "preds": preds,
    }


def random_test_image(checkpoint, split_csv=DEFAULT_TEST_CSV, dataset_dir="dataset"):
    df, class_names = load_split(split_csv)
    row = df.iloc[random.randrange(len(df))]
    image_path = image_dir_for_checkpoint(checkpoint, dataset_dir) / row["filename"]
    labels = [label for label in class_names if int(row[label]) == 1]
    return image_path, labels


def image_to_data_url(image_path):
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_bytes_to_pil(data):
    return Image.open(io.BytesIO(data))
