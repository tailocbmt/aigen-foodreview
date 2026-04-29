import torch
import numpy as np
import textstat
import re
import cv2
from copy import deepcopy


def multilabel_accuracy(targets, preds):
    """
    logits:  Tensor of shape [batch_size, num_labels]
    targets: Tensor of shape [batch_size, num_labels], values 0/1
    """
    y_pred = torch.tensor(deepcopy(preds))
    y_true = torch.tensor(deepcopy(targets))
    y_true = y_true.int()

    correct = (y_pred == y_true).float()
    return correct.mean().item()


def basic_counts(text):
    characters = len(text)
    words = len(text.split())
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    syllables = textstat.syllable_count(text)

    # complex words = ≥3 syllables
    complex_words = sum(1 for w in text.split()
                        if textstat.syllable_count(w) >= 3)

    return characters, words, sentences, syllables, complex_words


def text_metrics(text):
    c, w, s, sy, cw = basic_counts(text)

    return {
        "ARI": textstat.automated_readability_index(text),
        "DW": textstat.dale_chall_readability_score(text),
        "FR": textstat.flesch_reading_ease(text),
        "GFI": textstat.gunning_fog(text),
        "RT": textstat.reading_time(text, ms_per_char=14.69),
        "WPS": w / s
    }


def perplexity_gptneo(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    return torch.exp(loss).item()


def load_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def get_saliency_map(rgb):
    """
    Returns normalized saliency map in [0, 1].
    Requires opencv-contrib-python.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(bgr)

    if not success:
        raise RuntimeError("Saliency computation failed.")

    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 0)
    saliency_map = cv2.normalize(
        saliency_map, None, 0, 1, cv2.NORM_MINMAX
    )

    return saliency_map.astype(np.float32)


def get_foreground_mask(saliency_map):
    """
    Converts saliency map into foreground/background mask.
    """
    saliency_uint8 = (saliency_map * 255).astype(np.uint8)

    _, mask = cv2.threshold(
        saliency_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = mask > 0

    # fallback in case Otsu gives almost nothing
    if mask.mean() < 0.01:
        threshold = np.percentile(saliency_map, 85)
        mask = saliency_map >= threshold

    return mask


def salient_center(saliency_map):
    """
    Weighted center of saliency map.
    """
    h, w = saliency_map.shape
    y, x = np.indices((h, w))

    total = saliency_map.sum()

    if total == 0:
        return w / 2, h / 2

    cx = (x * saliency_map).sum() / total
    cy = (y * saliency_map).sum() / total

    return cx, cy


def diagonal_dominance(cx, cy, w, h):
    d1 = abs(h * cx - w * cy) / np.sqrt(h ** 2 + w ** 2)
    d2 = abs(h * cx + w * cy - w * h) / np.sqrt(h ** 2 + w ** 2)

    # normalized by image diagonal
    return min(d1, d2) / np.sqrt(w ** 2 + h ** 2)


def rule_of_thirds(cx, cy, w, h):
    points = [
        (w / 3, h / 3),
        (2 * w / 3, h / 3),
        (w / 3, 2 * h / 3),
        (2 * w / 3, 2 * h / 3),
    ]

    dists = [np.hypot(cx - x, cy - y) for x, y in points]

    # normalized by image diagonal
    return min(dists) / np.sqrt(w ** 2 + h ** 2)


def physical_visual_balance(rgb, saliency_map, n_segments=10):
    h, w, _ = rgb.shape

    segments = slic(
        rgb,
        n_segments=n_segments,
        compactness=10,
        start_label=0
    )

    centers = []
    weights = []

    for seg_id in np.unique(segments):
        mask = segments == seg_id
        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        cx = xs.mean()
        cy = ys.mean()
        weight = saliency_map[mask].mean() * mask.sum()

        centers.append((cx, cy))
        weights.append(weight)

    centers = np.array(centers)
    weights = np.array(weights)

    if weights.sum() == 0:
        weighted_x = w / 2
        weighted_y = h / 2
    else:
        weighted_x = np.sum(centers[:, 0] * weights) / weights.sum()
        weighted_y = np.sum(centers[:, 1] * weights) / weights.sum()

    horizontal_balance = abs(weighted_x - w / 2) / w
    vertical_balance = abs(weighted_y - h / 2) / h

    return horizontal_balance, vertical_balance


def horizontal_color_balance(rgb_float):
    h, w, _ = rgb_float.shape
    mid = w // 2

    left = rgb_float[:, :mid, :]
    right = rgb_float[:, w - mid:, :]

    right_flipped = np.flip(right, axis=1)

    diff = np.linalg.norm(left - right_flipped, axis=2)

    return np.mean(diff)


def vertical_color_balance(rgb_float):
    h, w, _ = rgb_float.shape
    mid = h // 2

    top = rgb_float[:mid, :, :]
    bottom = rgb_float[h - mid:, :, :]

    bottom_flipped = np.flip(bottom, axis=0)

    diff = np.linalg.norm(top - bottom_flipped, axis=2)

    return np.mean(diff)


def size_difference(mask):
    total = mask.size
    fg = mask.sum()
    bg = total - fg

    return (bg - fg) / total


def foreground_background_color_difference(rgb_float, mask):
    fg_pixels = rgb_float[mask]
    bg_pixels = rgb_float[~mask]

    if len(fg_pixels) == 0 or len(bg_pixels) == 0:
        return 0.0

    mean_fg = fg_pixels.mean(axis=0)
    mean_bg = bg_pixels.mean(axis=0)

    return np.linalg.norm(mean_fg - mean_bg)


def texture_difference(rgb, mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edges = edges > 0

    fg_edges = edges[mask]
    bg_edges = edges[~mask]

    if len(fg_edges) == 0 or len(bg_edges) == 0:
        return 0.0

    fg_density = fg_edges.mean()
    bg_density = bg_edges.mean()

    return abs(fg_density - bg_density)


def image_metrics(image_path):
    # Read image as RGB
    rgb = load_image(image_path)
    rgb_float = rgb.astype(np.float32) / 255.0

    h, w, _ = rgb.shape

    # Convert to HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]   # OpenCV float HSV hue: 0–360
    S = hsv[:, :, 1]   # 0–1
    V = hsv[:, :, 2]   # 0–1

    # Attribute 1: Brightness
    brightness = np.mean(V)

    # Attribute 2: Contrast
    contrast = np.std(V)

    # Attribute 3: Saturation
    saturation = np.mean(S)

    # Attribute 4: Clarity
    clarity = np.mean(V > 0.7)

    # Attribute 5: Warm
    warm = np.mean((H < 60) | (H > 220))

    # Attribute 6: Colorfulness, Hasler & Suesstrunk
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    rg = R - G
    yb = 0.5 * (R + G) - B

    sigma_rg_yb = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mu_rg_yb = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    colorfulness = sigma_rg_yb + 0.3 * mu_rg_yb

    saliency_map = get_saliency_map(rgb)
    mask = get_foreground_mask(saliency_map)
    cx, cy = salient_center(saliency_map)

    horizontal_physical_balance, vertical_physical_balance = (
        physical_visual_balance(rgb, saliency_map)
    )

    metrics = {
        # Attributes 1–6
        "BRI": float(np.mean(V)),
        "CON": float(np.std(V)),
        "SAT": float(np.mean(S)),
        "CLA": float(np.mean(V > 0.7)),
        "WAR": float(np.mean((H < 60) | (H > 220))),
        "COL": float(colorfulness),

        # Attributes 7–8
        "DD": float(diagonal_dominance(cx, cy, w, h)),
        "ROT": float(rule_of_thirds(cx, cy, w, h)),

        # Attributes 9–10
        "HPVB": float(horizontal_physical_balance),
        "VPVB": float(vertical_physical_balance),

        # Attributes 11–12
        "HCVC": float(horizontal_color_balance(rgb_float)),
        "VCVC": float(vertical_color_balance(rgb_float)),

        # Attributes 13–15
        "SD": float(size_difference(mask)),
        "CD": float(
            foreground_background_color_difference(rgb_float, mask)
        ),
        "TD": float(texture_difference(rgb, mask)),
    }

    return metrics
