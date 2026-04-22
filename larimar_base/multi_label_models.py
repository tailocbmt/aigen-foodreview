import torch
import torch.nn as nn
from transformers import BertModel
from transformers import ResNetForImageClassification


class FakeNewsMultimodal(nn.Module):
    def __init__(self, output_dim: int = 2):
        super().__init__()

        # Text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size  # 768

        # Image encoder (Hugging Face)
        self.image_encoder = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=2,
            ignore_mismatched_sizes=True
        )

        # Remove classification head → get features
        self.image_encoder.classifier = nn.Identity()
        image_dim = 2048

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, inputs):
        images = inputs['pixel_values']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # Text
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Image
        image_outputs = self.image_encoder(images)
        image_features = image_outputs.logits  # now 2048-dim

        # Fusion
        rint("text_features.shape:", text_features.shape)
        print("image_features.shape:", image_features.shape)
        combined = torch.cat((text_features, image_features), dim=1)

        logits = self.classifier(combined)
        return logits


class FakeNewsMultimodalSplit(nn.Module):
    def __init__(self, output_dim: int = 2):
        super().__init__()

        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size  # 768

        self.image_encoder = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        self.image_encoder.classifier = nn.Identity()
        image_dim = 2048

        # Separate heads
        self.text_head = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.image_head = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # Optional fusion head if you want cross-modal interaction
        self.fusion_head = nn.Sequential(
            nn.Linear(text_dim + image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, input_ids, attention_mask, images):
        # Text
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Image
        image_outputs = self.image_encoder(images)
        image_features = image_outputs.logits  # now 2048-dim

        combined = torch.cat([text_features, image_features], dim=1)

        text_logit = self.text_head(text_features)
        image_logit = self.image_head(image_features)
        fused_logits = self.fusion_head(combined)

        return {
            "text_logit": text_logit,
            "image_logit": image_logit,
            "fused_logits": fused_logits
        }
