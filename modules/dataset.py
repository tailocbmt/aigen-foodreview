import json

from PIL import Image, UnidentifiedImageError
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
import os
from PIL import Image
from larimar_base.utils import process_dct_img
from modules.utils import DatasetTransforms
from raid.utils import load_data


@dataclass
class DatasetConfig:
    file: str
    real_image_dir: str
    fake_image_dir: str
    max_length: int
    fake_text_columns: List[str] = field(default_factory=lambda: [
        "qwen_rewritten_title",
        "llama_rewritten_title",
        "mistral_rewritten_title",
    ])
    fake_image_columns: List[str] = field(default_factory=lambda: [
        "sd_img_path",
        "flux_img_path",
        "z_img_path",
        "qwen_img_path",
    ])
    mode: str = "train"
    return_combo_name: bool = False

    def __post_init__(self):
        if self.mode not in ("train", "val", "test"):
            raise ValueError(
                f"mode must be 'train', 'val', or 'test', got '{self.mode}'")


# ── Change 2: Named combo tuple instead of bare ("real","real") strings ──
@dataclass(frozen=True)
class Combo:
    text_mode: str   # "real" | "fake"
    image_mode: str  # "real" | "fake"


_COMBOS = [
    Combo("real", "real"),
    Combo("fake", "real"),
    Combo("real", "fake"),
    Combo("fake", "fake"),
]

_COMBO_LABELS: Dict[Tuple[int, int], str] = {
    (0, 0): "real_text_real_img",
    (1, 0): "fake_text_real_img",
    (0, 1): "real_text_fake_img",
    (1, 1): "fake_text_fake_img",
}


class TextDataset(Dataset):
    def __init__(self, file, tokenizer, max_length: int = None):
        super().__init__()
        self.file = file
        self.data = pd.read_csv(self.file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        current = self.data.iloc[idx]
        text = current.text
        label = current.label
        if self.max_length:
            encoded_input = self.tokenizer(
                text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        else:
            encoded_input = self.tokenizer(text, return_tensors='pt')
        output = {'input': encoded_input, 'label': label}
        return output


class VisionDataset(Dataset):
    def __init__(self, file, image_dir, transform=None):
        super().__init__()
        self.file = file
        self.data = pd.read_csv(file)
        # /home/ubuntu/combat-ai-restaurants/multimodal-dataset/data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        image_id = item.ID
        label = item.label
        if label == 0:
            image_path = os.path.join(
                self.image_dir, str(image_id) + '.jpg')
        elif label == 1:
            image_path = os.path.join(
                self.image_dir, str(image_id) + '.jpg')
        else:
            pass
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image, return_tensors="pt")
        return {'input': image, 'label': label}


class MultimodalDataset(Dataset):
    def __init__(self, file, image_dir, processor, max_length):
        super().__init__()
        self.file = file
        self.data = pd.read_csv(file)
        # /home/ubuntu/combat-ai-restaurants/multimodal-dataset/data
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        image_id = item.ID
        label = item.label
        text = item.text
        if label == 0:
            image_path = os.path.join(self.image_dir, str(image_id) + '.jpg')
        elif label == 1:
            image_path = os.path.join(self.image_dir, str(image_id) + '.jpg')
        else:
            pass
        image = Image.open(image_path).convert('RGB')
        inputs = self.tokenize(text=[text], images=[image])
        return {'inputs': inputs, 'label': label}

    def tokenize(self, text: list, images: list):
        inputs = self.processor(text=text, images=images, return_tensors="pt",
                                max_length=self.max_length, truncation=True, padding="max_length")
        return inputs


class HintsOfTruthMultimodalDataset(Dataset):
    def __init__(self, file, image_dir, split, processor, max_length):
        super().__init__()
        self.file = file
        self.real_data = load_dataset("michiel/hints_of_truth", split=split)

        self.fake_data = pd.read_csv(file)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        self.real_len = len(self.real_data)
        self.fake_len = len(
            self.fake_data) if self.fake_data is not None else 0
        self.total_len = self.real_len + self.fake_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < self.real_len:
            row = self.real_data[index]
            text = str(row["text"])
            image = row["image"].convert('RGB')
            label = 1
        else:
            row = self.fake_data.iloc[index - self.real_len]
            text = str(row["llava_caption"])
            image_path = os.path.join(self.image_dir, row["saved_image_path"])
            image = Image.open(image_path).convert('RGB')
            label = 0

        inputs = self.tokenize(text=[text], images=[image])

        return {'inputs': inputs, 'label': label}

    def tokenize(self, text: list, images: list):
        inputs = self.processor(text=text, images=images, return_tensors="pt",
                                max_length=self.max_length, truncation=True, padding="max_length")
        return inputs


class HintsOfTruthTextDataset(HintsOfTruthMultimodalDataset):
    def __init__(self, file, split, tokenizer, max_length: int = None):
        super().__init__(
            file=file,
            image_dir=None,
            split=split,
            processor=None,
            max_length=max_length
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if idx < self.real_len:
            row = self.real_data[idx]
            text = str(row["text"])
            label = 1
        else:
            row = self.fake_data.iloc[idx - self.real_len]
            text = str(row["llava_caption"])
            label = 0

        if self.max_length:
            encoded_input = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
        else:
            encoded_input = self.tokenizer(text, return_tensors="pt")

        output = {"input": encoded_input, "label": label}
        return output


class HintsOfTruthVisionDataset(HintsOfTruthMultimodalDataset):
    def __init__(self, file, image_dir, split, transform=None):
        super().__init__(
            file=file,
            image_dir=image_dir,
            split=split,
            processor=None,
            max_length=None
        )
        self.transform = transform

    def __getitem__(self, index):
        if index < self.real_len:
            item = self.real_data[index]
            image = item["image"].convert("RGB")
            label = 1
        else:
            item = self.fake_data.iloc[index - self.real_len]
            image_path = os.path.join(self.image_dir, item["saved_image_path"])
            image = Image.open(image_path).convert("RGB")
            label = 0

        if self.transform:
            image = self.transform(image, return_tensors="pt")

        return {"input": image, "label": label}


class EvonsMultimodalDataset(Dataset):
    def __init__(self, file, real_image_dir, fake_image_dir, processor, max_length):
        super().__init__()
        self.file = file

        self.real_data = pd.read_csv(file)
        self.fake_data = deepcopy(self.real_data)
        self.fake_data['is_fake'] = 1
        self.data = pd.concat(
            [self.real_data, self.fake_data], ignore_index=True)

        self.real_image_dir = real_image_dir
        self.fake_image_dir = fake_image_dir
        self.processor = processor
        self.max_length = max_length

        self.total_len = len(self.data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        item = self.data.iloc[index]
        label = item.is_fake
        if label == 0:
            image_name = item.image_fn
            media_source = item.media_source
            text = item.real_text
            image_path = os.path.join(
                self.real_image_dir, media_source, str(image_name))
        else:
            image_name = item.fake_img_paths
            text = item.fake_text
            image_path = os.path.join(self.fake_image_dir, str(image_name))

        image = Image.open(image_path).convert('RGB')
        inputs = self.tokenize(text=[text], images=[image])
        return {'inputs': inputs, 'label': label}

    def tokenize(self, text: list, images: list):
        inputs = self.processor(text=text, images=images, return_tensors="pt",
                                max_length=self.max_length, truncation=True, padding="max_length")
        return inputs


class EvonsOnlineMultimodalDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, processor):
        super().__init__()
        self.cfg = cfg
        self.processor = processor

        # ── Change 3: Validate file exists early, not at first __getitem__ ──
        csv_path = Path(cfg.file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        self.data = pd.read_csv(csv_path).reset_index(drop=True)

        # ── Change 4: Validate required columns at init time ──
        self._validate_columns()

    # ── Change 5: Column validation extracted to its own method ──
    def _validate_columns(self) -> None:
        required = {"media_source"}
        text_real = {"real_title", "real_text"}
        image_real = {"real_img_path", "image_fn"}

        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not (text_real & set(self.data.columns)):
            raise ValueError(f"Need at least one of {text_real} in CSV.")
        if not (image_real & set(self.data.columns)):
            raise ValueError(f"Need at least one of {image_real} in CSV.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data.iloc[index]

        fake_text_pool = self._get_fake_text_pool(row)
        fake_image_pool = self._get_fake_image_pool(row)

        combo = self._sample_combo(index)

        text, text_generator, text_label = self._sample_text(
            row, fake_text_pool, combo.text_mode, index
        )
        image_path, image_generator, image_label = self._sample_image(
            row, fake_image_pool, combo.image_mode, index
        )

        # ── Change 6: Catch corrupt images gracefully instead of crashing ──
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            raise RuntimeError(
                f"Failed to load image at index {index}: {image_path}"
            ) from e

        inputs = self.tokenize(text=[text], images=[image])
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        label = torch.tensor([text_label, image_label], dtype=torch.float)

        output: Dict[str, Any] = {"inputs": inputs, "label": label}

        if self.cfg.return_combo_name:
            output.update({
                "combo_label": _COMBO_LABELS[(text_label, image_label)],
                "text_generator": text_generator,
                "image_generator": image_generator,
                "source_index": index,
            })

        return output

    def _sample_combo(self, index: int) -> Combo:
        if self.cfg.mode == "train":
            return random.choice(_COMBOS)
        # Deterministic for val/test — covers all 4 combos evenly
        return _COMBOS[index % len(_COMBOS)]

    def _get_fake_text_pool(self, row: pd.Series) -> List[Tuple[str, str]]:
        pool = [
            (col, str(row[col]).strip())
            for col in self.cfg.fake_text_columns
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip()
        ]
        if not pool:
            raise ValueError(
                f"No valid fake text found. Checked columns: {self.cfg.fake_text_columns}"
            )
        return pool

    def _get_fake_image_pool(self, row: pd.Series) -> List[Tuple[str, str]]:
        pool = [
            (col, str(row[col]).strip())
            for col in self.cfg.fake_image_columns
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip()
        ]
        if not pool:
            raise ValueError(
                f"No valid fake image found. Checked columns: {self.cfg.fake_image_columns}"
            )
        return pool

    def _sample_text(
        self,
        row: pd.Series,
        fake_text_pool: List[Tuple[str, str]],
        mode: str,
        index: int,
    ) -> Tuple[str, str, int]:
        if mode == "real":
            for col in ("real_title", "real_text"):
                if col in row.index and pd.notna(row[col]):
                    return str(row[col]), "real", 0
            raise ValueError(
                "No real text column found. Expected 'real_title' or 'real_text'.")

        # ── Change 7: DRY — single branch for fake sampling (train vs val) ──
        pool_index = (
            random.randrange(len(fake_text_pool))
            if self.cfg.mode == "train"
            else index % len(fake_text_pool)
        )
        gen_name, text = fake_text_pool[pool_index]
        return text, gen_name, 1

    def _sample_image(
        self,
        row: pd.Series,
        fake_image_pool: List[Tuple[str, str]],
        mode: str,
        index: int,
    ) -> Tuple[str, str, int]:
        if mode == "real":
            media_source = (
                str(row["media_source"]).strip()
                if pd.notna(row.get("media_source", float("nan")))
                else ""
            )
            for col in ("real_img_path", "image_fn"):
                if col in row.index and pd.notna(row[col]):
                    image_name = str(row[col])
                    # ── Change 8: Use pathlib.Path for cross-platform safety ──
                    image_path = (
                        Path(self.cfg.real_image_dir) /
                        media_source / image_name
                        if media_source
                        else Path(self.cfg.real_image_dir) / image_name
                    )
                    return str(image_path), "real", 0
            raise ValueError(
                "No real image column found. Expected 'real_img_path' or 'image_fn'.")

        pool_index = (
            random.randrange(len(fake_image_pool))
            if self.cfg.mode == "train"
            else index % len(fake_image_pool)
        )
        gen_name, image_name = fake_image_pool[pool_index]
        image_path = Path(self.cfg.fake_image_dir) / image_name
        return str(image_path), gen_name, 1

    def tokenize(self, text: List[str], images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        return self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            max_length=self.cfg.max_length,
            truncation=True,
            padding="max_length",
        )


class EvonsOfflineTextDataset(Dataset):
    def __init__(self, file, processor, max_length):
        super().__init__()
        self.data = pd.read_csv(file)

        self.tokenizer = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _build_text(self, item):
        title = "" if pd.isna(item["title"]) else str(item["title"])
        description = "" if pd.isna(
            item["description"]) else str(item["description"])

        if title and description:
            return f"{title} {description}"
        return title or description

    def __getitem__(self, index):
        item = self.data.iloc[index]

        text = self._build_text(item)

        label = int(item["label_text"])

        if self.max_length:
            encoded_input = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
        else:
            encoded_input = self.tokenizer(text, return_tensors="pt")

        return {
            "input": encoded_input,
            "label": label,
        }


class EvonsOfflineVisionDataset(Dataset):
    def __init__(self, file, image_dir, processor):
        super().__init__()
        self.data = pd.read_csv(file)

        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        image_rel_path = item["image_path"]
        image_path = os.path.join(self.image_dir, image_rel_path)

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")

        labels = int(item["label_image"])

        return {
            "input": inputs,
            "label": labels,
        }


class EvonsOfflineMultimodalDataset(Dataset):
    def __init__(self, file, image_dir, processor, max_length):
        super().__init__()
        self.data = pd.read_csv(file)

        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        if 'val' in file:
            self.mode = 'val'
        elif 'train' in file:
            self.mode = 'train'
        else:
            self.mode = 'test'

        if self.transform_image is None:
            transform_pipeline = DatasetTransforms(
                input_size=224,
                mode=self.mode
            )
            self.transform_image = transform_pipeline.get_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        text = self._build_text(item)

        image_rel_path = item["image_path"]
        image_path = os.path.join(self.image_dir, image_rel_path)

        image = Image.open(image_path).convert("RGB")
        inputs = self.tokenize(text=[text], images=[image])

        if self.transform_image is not None:
            image_np = np.array(image)  # Shape: (H, W, C), dtype: uint8
            image_tensor = self.transform_image(image=image_np)['image']
            inputs["pixel_values"] = image_tensor

        labels = torch.tensor(
            [
                int(item["label_text"]),
                int(item["label_image"]),
            ],
            dtype=torch.float
        )

        return {
            "inputs": inputs,
            "label": labels,
        }

    def _build_text(self, item):
        title = "" if pd.isna(item["title"]) else str(item["title"])
        description = "" if pd.isna(
            item["description"]) else str(item["description"])

        if title and description:
            return f"{title} {description}"
        return title or description

    def tokenize(self, text: list, images: list):
        if isinstance(self.processor, list):
            image_inputs = self.processor[0](
                images=images, return_tensors="pt")

            if self.max_length:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )
            else:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                )

            return {
                **image_inputs,
                **text_inputs,
            }
        else:
            return self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )


class EvonsOfflineMultimodalWDctDataset(Dataset):
    def __init__(
        self,
        file,
        image_dir,
        processor,
        max_length,
        transform_image=None,
        transform_dct=None,
    ):
        super().__init__()

        self.data = pd.read_csv(file)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        if 'val' in file:
            self.mode = 'val'
        elif 'train' in file:
            self.mode = 'train'
        else:
            self.mode = 'test'

        if transform_image is None:
            transform_pipeline = DatasetTransforms(
                input_size=224,
                mode=self.mode
            )
            self.transform_image = transform_pipeline.get_transforms()
        if transform_dct is None:
            self.transform_dct = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        text = self._build_text(item)

        image_rel_path = item["image_path"]
        image_path = os.path.join(self.image_dir, image_rel_path)

        image = Image.open(image_path).convert("RGB")

        inputs = self.tokenize(text=[text], images=[image])

        # Standard image tensor for image encoder
        if self.transform_image is not None:
            image_np = np.array(image)  # Shape: (H, W, C), dtype: uint8
            image_tensor = self.transform_image(image=image_np)['image']
            inputs["image"] = image_tensor

        # DCT image tensor for frequency branch
        if self.transform_dct is not None:
            gray_image = image.convert("L")
            dct_img = self.transform_dct(gray_image)
            dct_img = process_dct_img(dct_img)
            inputs["dct_img"] = dct_img

        labels = torch.tensor(
            [
                int(item["label_text"]),
                int(item["label_image"]),
            ],
            dtype=torch.float
        )

        return {
            "inputs": inputs,
            "label": labels,
        }

    def _build_text(self, item):
        title = "" if pd.isna(item["title"]) else str(item["title"])
        description = "" if pd.isna(
            item["description"]) else str(item["description"])

        if title and description:
            return f"{title} {description}"
        return title or description

    def tokenize(self, text: list, images: list):
        if isinstance(self.processor, (list, tuple)):
            image_inputs = self.processor[0](
                images=images,
                return_tensors="pt"
            )

            if self.max_length:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )
            else:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                )

            return {
                **image_inputs,
                **text_inputs,
            }

        else:
            return self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )


class AIGenFoodMultimodalDataset(Dataset):
    def __init__(self, file, image_dir, processor, max_length):
        super().__init__()
        self.data = pd.read_csv(file)

        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        if 'val' in file:
            self.mode = 'val'
        elif 'train' in file:
            self.mode = 'train'
        else:
            self.mode = 'test'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        text = str(item["text"])

        image_path = os.path.join(self.image_dir, str(item["ID"])) + '.jpg'

        image = Image.open(image_path).convert("RGB")
        inputs = self.tokenize(text=[text], images=[image])

        labels = torch.tensor(
            [
                int(item["label"]),
            ],
            dtype=torch.float
        )

        return {
            "inputs": inputs,
            "label": labels,
        }

    def tokenize(self, text: list, images: list):
        if isinstance(self.processor, list):
            image_inputs = self.processor[0](
                images=images, return_tensors="pt")

            if self.max_length:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )
            else:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                )

            return {
                **image_inputs,
                **text_inputs,
            }
        else:
            return self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )


class RAIDMultimodalDataset(Dataset):
    def __init__(self, processor, max_length):
        super().__init__()
        self.data = load_data(split="test")

        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        item_id = item["id"]

        text = str(item["generation"])

        image = Image.new("RGB", (224, 224), color=(0, 0, 0))          # black
        inputs = self.tokenize(text=[text], images=[image])

        labels = torch.tensor(
            [
                1,
                1
            ],
            dtype=torch.float
        )

        return {
            "ids": item_id,
            "inputs": inputs,
            "label": labels,
        }

    def tokenize(self, text: list, images: list):
        if isinstance(self.processor, list):
            image_inputs = self.processor[0](
                images=images, return_tensors="pt")

            if self.max_length:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )
            else:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                )

            return {
                **image_inputs,
                **text_inputs,
            }
        else:
            return self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )


class SemEval2024MultimodalDataset(Dataset):
    def __init__(self, processor, max_length):
        super().__init__()
        self.data = []
        file_path = "../semeval2024/SubtaskA/subtaskA_monolingual.jsonl"

        if os.path.exists(file_path):
            # lines=True is required for .jsonl files
            self.data = pd.read_json(file_path, lines=True)
            print(
                f"Successfully loaded DataFrame with shape: {self.data.shape}")
        else:
            print(f"Error: {file_path} not found in the current directory.")

        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        item_id = item["id"]

        text = str(item["text"])

        image = Image.new("RGB", (224, 224), color=(0, 0, 0))          # black
        inputs = self.tokenize(text=[text], images=[image])

        labels = torch.tensor(
            [
                item["label"]
            ],
            dtype=torch.float
        )

        return {
            "ids": item_id,
            "inputs": inputs,
            "label": labels,
        }

    def tokenize(self, text: list, images: list):
        if isinstance(self.processor, list):
            image_inputs = self.processor[0](
                images=images, return_tensors="pt")

            if self.max_length:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )
            else:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                )

            return {
                **image_inputs,
                **text_inputs,
            }
        else:
            return self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )


class DefactifyMultimodalDataset(Dataset):
    def __init__(self, processor, max_length):
        super().__init__()
        self.data = load_dataset(
            "Rajarshi-Roy-research/Defactify_Image_Dataset", split="test")

        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        text = str(item["Caption"])
        image = item["Image"].convert("RGB")

        inputs = self.tokenize(text=[text], images=[image])

        labels = torch.tensor(
            [
                int(item["Label_A"]),
            ],
            dtype=torch.float
        )

        return {
            "inputs": inputs,
            "label": labels,
        }

    def tokenize(self, text: list, images: list):
        if isinstance(self.processor, list):
            image_inputs = self.processor[0](
                images=images, return_tensors="pt")

            if self.max_length:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )
            else:
                text_inputs = self.processor[1](
                    text,
                    return_tensors="pt",
                )

            return {
                **image_inputs,
                **text_inputs,
            }
        else:
            return self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
