from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
import os
from PIL import Image


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
