from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPImageProcessor


class HintsOfTruthRealDataset(Dataset):
    """
    Real samples from Hugging Face dataset: michiel/hints_of_truth

    We ignore the dataset's original 'labels' field and instead create:
        label = 1  (real)
    """

    def __init__(
        self,
        split: str,
        tokenizer_name: str,
        vision_model_name: str,
        max_text_length: int = 128,
    ):
        super().__init__()
        self.split = split
        self.max_text_length = max_text_length

        self.dataset = load_dataset("michiel/hints_of_truth", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            vision_model_name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.dataset[idx]

        text = str(row["text"])
        image = row["image"]

        text_enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        image_enc = self.image_processor(
            images=image,
            return_tensors="pt",
        )

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": image_enc["pixel_values"].squeeze(0),
            "labels": torch.tensor(1, dtype=torch.long),  # REAL
        }


class LocalFakeCSVDataset(Dataset):
    """
    Fake samples from local CSV + image folder.

    Required CSV columns:
        - llava_caption
        - saved_image_path

    Optional:
        - if CSV has labels, they are ignored here because this dataset is fake-only

    Output label:
        label = 0  (fake)
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        tokenizer_name: str,
        vision_model_name: str,
        max_text_length: int = 128,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.image_root = Path(image_root)
        self.max_text_length = max_text_length

        self.df = pd.read_csv(csv_path)

        required_cols = ["llava_caption", "saved_image_path"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"Missing required column '{col}' in {csv_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            vision_model_name)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_path_value: str) -> Path:
        path = Path(str(image_path_value))
        if path.is_absolute():
            return path
        return self.image_root / path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        text = str(row["llava_caption"])
        image_path = self._resolve_image_path(row["saved_image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        text_enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        image_enc = self.image_processor(
            images=image,
            return_tensors="pt",
        )

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": image_enc["pixel_values"].squeeze(0),
            "labels": torch.tensor(0, dtype=torch.long),  # FAKE
        }


class CombinedRealFakeDataset(Dataset):
    """
    Combine one real dataset and one fake dataset into a single dataset.
    """

    def __init__(self, real_dataset: Dataset, fake_dataset: Optional[Dataset] = None):
        super().__init__()
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset

        self.real_len = len(real_dataset)
        self.fake_len = len(fake_dataset) if fake_dataset is not None else 0
        self.total_len = self.real_len + self.fake_len

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.real_len:
            return self.real_dataset[idx]
        if self.fake_dataset is None:
            raise IndexError("Fake dataset is None but fake index requested.")
        return self.fake_dataset[idx - self.real_len]


class MultimodalHintsOfTruthDataModule:
    """
    DataModule for:
      - real data from Hugging Face michiel/hints_of_truth
      - fake data from local CSV + local image folders

    Suggested split mapping:
      - train = dev1(real) + fake_dev1
      - val   = dev2(real) + fake_dev2
      - test  = test(real) + optional fake_test

    If fake_test is not provided, test loader will contain real-only test data.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

    def setup(self, max_samples: Optional[int] = None):
        data_cfg = self.config

        tokenizer_name = data_cfg.get("tokenizer_name", "bert-base-uncased")
        vision_model_name = data_cfg.get(
            "vision_model_name", "openai/clip-vit-base-patch32")
        max_text_length = data_cfg.get("max_text_length", 128)

        # -------------------------
        # Real data from HF
        # -------------------------
        real_train = HintsOfTruthRealDataset(
            split=data_cfg.get("real_train_split", "dev1"),
            tokenizer_name=tokenizer_name,
            vision_model_name=vision_model_name,
            max_text_length=max_text_length,
        )

        real_val = HintsOfTruthRealDataset(
            split=data_cfg.get("real_val_split", "dev2"),
            tokenizer_name=tokenizer_name,
            vision_model_name=vision_model_name,
            max_text_length=max_text_length,
        )

        real_test = HintsOfTruthRealDataset(
            split=data_cfg.get("real_test_split", "test"),
            tokenizer_name=tokenizer_name,
            vision_model_name=vision_model_name,
            max_text_length=max_text_length,
        )

        # -------------------------
        # Local fake data
        # -------------------------
        fake_train_csv = data_cfg.get("fake_train_csv")
        fake_train_image_root = data_cfg.get("fake_train_image_root")

        fake_val_csv = data_cfg.get("fake_val_csv")
        fake_val_image_root = data_cfg.get("fake_val_image_root")

        fake_test_csv = data_cfg.get("fake_test_csv")
        fake_test_image_root = data_cfg.get("fake_test_image_root")

        fake_train = None
        fake_val = None
        fake_test = None

        if fake_train_csv and fake_train_image_root:
            fake_train = LocalFakeCSVDataset(
                csv_path=fake_train_csv,
                image_root=fake_train_image_root,
                tokenizer_name=tokenizer_name,
                vision_model_name=vision_model_name,
                max_text_length=max_text_length,
            )

        if fake_val_csv and fake_val_image_root:
            fake_val = LocalFakeCSVDataset(
                csv_path=fake_val_csv,
                image_root=fake_val_image_root,
                tokenizer_name=tokenizer_name,
                vision_model_name=vision_model_name,
                max_text_length=max_text_length,
            )

        if fake_test_csv and fake_test_image_root:
            fake_test = LocalFakeCSVDataset(
                csv_path=fake_test_csv,
                image_root=fake_test_image_root,
                tokenizer_name=tokenizer_name,
                vision_model_name=vision_model_name,
                max_text_length=max_text_length,
            )

        # -------------------------
        # Combine real + fake
        # -------------------------
        self.train_dataset = CombinedRealFakeDataset(real_train, fake_train)
        val_dataset = CombinedRealFakeDataset(real_val, fake_val)
        test_dataset = CombinedRealFakeDataset(real_test, fake_test)

        # Optional max_samples for quick debugging
        if max_samples is not None:
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset,
                list(range(min(max_samples, len(self.train_dataset))))
            )
            val_dataset = torch.utils.data.Subset(
                val_dataset,
                list(range(min(max_samples, len(val_dataset))))
            )
            test_dataset = torch.utils.data.Subset(
                test_dataset,
                list(range(min(max_samples, len(test_dataset))))
            )

        self.val_datasets = [val_dataset]
        self.test_datasets = [test_dataset]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 8),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
            drop_last=self.config.get("drop_last", False),
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.config.get("batch_size", 8),
                shuffle=False,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=self.config.get("pin_memory", True),
                drop_last=False,
            )
            for ds in self.val_datasets
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.config.get("batch_size", 8),
                shuffle=False,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=self.config.get("pin_memory", True),
                drop_last=False,
            )
            for ds in self.test_datasets
        ]


def create_data_module(config: Dict) -> MultimodalHintsOfTruthDataModule:
    return MultimodalHintsOfTruthDataModule(config)
