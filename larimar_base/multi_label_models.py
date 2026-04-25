import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import ResNetForImageClassification
from larimar_base.base_models import BaseDetector, MemoryAugmentedDetector
from larimar_base.utils import DctCNN, PositionalWiseFeedForward, multimodal_fusion_layer


class FakeNewsMultimodal(BaseDetector):
    def __init__(self, output_dim: int = 2):
        super().__init__()

        # Text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_dim = self.text_encoder.config.hidden_size  # 768

        # Image encoder (Hugging Face)
        self.image_encoder = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=2,
            ignore_mismatched_sizes=True
        )

        # Remove classification head → get features
        self.image_encoder.classifier = nn.Identity()
        self.image_dim = 2048

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.text_dim + self.image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def fusion_block(self, text, image):
        combined = torch.cat((text, image), dim=1)
        return combined

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
        image_features = torch.flatten(image_features, start_dim=1)

        # Fusion
        combined = self.fusion_block(text=text_features, image=image_features)

        logits = self.classifier(combined)
        return logits


class FakeNewsMultimodalWMemory(MemoryAugmentedDetector):
    def __init__(
        self,
        out_dim=1,
        use_memory=True,
        memory_size=512,
        memory_mode="read_write",
        fusion_type="add"
    ):
        super().__init__(
            feature_dim=1024,
            out_dim=out_dim,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )

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
            nn.Linear(512, out_dim)
        )

    def feature_extractor(self, inputs):
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
        image_features = torch.flatten(image_features, start_dim=1)

        # Fusion
        combined = torch.cat((text_features, image_features), dim=1)

        return combined


class CoAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.5):
        super().__init__()
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # self.text_norm = nn.LayerNorm(dim)
        # self.image_norm = nn.LayerNorm(dim)
        self.feed_forward_1 = PositionalWiseFeedForward(
            dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(
            dim, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feats, image_feats):
        """
        text_feats:  [B, 1, D]
        image_feats: [B, 1, D]
        """

        # text attends to image
        text_attended, _ = self.text_to_image_attn(
            query=text_feats,
            key=image_feats,
            value=image_feats
        )
        # text_out = self.text_norm(text_feats + self.dropout(text_attended))
        text_out = self.feed_forward_1(text_attended)

        # image attends to text
        image_attended, _ = self.image_to_text_attn(
            query=image_feats,
            key=text_feats,
            value=text_feats
        )
        image_out = self.feed_forward_2(text_attended)
        # image_out = self.image_norm(image_feats + self.dropout(image_attended))

        return text_out, image_out


class FakeNewsMultimodalCoAttention(FakeNewsMultimodal):
    def __init__(
        self,
        output_dim: int = 2,
        fusion_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.5
    ):
        super().__init__(output_dim=output_dim)

        # Project to same dimension for co-attention
        self.text_proj = nn.Linear(self.text_dim, fusion_dim)
        self.image_proj = nn.Linear(self.image_dim, fusion_dim)

        # Co-attention
        self.co_attention = CoAttentionBlock(
            dim=fusion_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def fusion_block(self, text, image):
        # Project to shared space
        text_features = self.text_proj(text)    # [B, fusion_dim]
        image_features = self.image_proj(image)  # [B, fusion_dim]

        # Add sequence dimension for MultiheadAttention
        text_features = text_features.unsqueeze(1)   # [B, 1, D]
        image_features = image_features.unsqueeze(1)  # [B, 1, D]

        # Co-attention
        text_co, image_co = self.co_attention(text_features, image_features)

        # Remove sequence dimension
        text_co = text_co.squeeze(1)     # [B, D]
        image_co = image_co.squeeze(1)   # [B, D]

        # Fusion
        combined = torch.cat((text_co, image_co), dim=1)  # [B, 2D]

        return combined


class FakeNewsMultimodalWMemoryCoAttention(FakeNewsMultimodal):
    def __init__(
            self,
            out_dim: int = 2,
            use_memory: bool = True,
            memory_size: int = 512,
            memory_mode: str = "read_write",
            fusion_type: str = "concat",
            proj_dim: int = 256,
            num_heads: int = 8,
            attn_dropout: float = 0.5,
            fusion_dim: int = 512):

        super().__init__(
            feature_dim=proj_dim * 2,
            out_dim=out_dim,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )

        # Project to same dimension for co-attention
        self.text_proj = nn.Linear(self.text_dim, fusion_dim)
        self.image_proj = nn.Linear(self.image_dim, fusion_dim)

        # Co-attention
        self.co_attention = CoAttentionBlock(
            dim=proj_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def fusion_block(self, text, image):
        # Project to shared space
        text_features = self.text_proj(text)    # [B, fusion_dim]
        image_features = self.image_proj(image)  # [B, fusion_dim]

        # Add sequence dimension for MultiheadAttention
        text_features = text_features.unsqueeze(1)   # [B, 1, D]
        image_features = image_features.unsqueeze(1)  # [B, 1, D]

        # Co-attention
        text_co, image_co = self.co_attention(text_features, image_features)

        # Remove sequence dimension
        text_co = text_co.squeeze(1)     # [B, D]
        image_co = image_co.squeeze(1)   # [B, D]

        # Fusion
        combined = torch.cat((text_co, image_co), dim=1)  # [B, 2D]

        return combined

    def feature_extractor(self, inputs):
        images = inputs['pixel_values']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Image
        image_outputs = self.image_encoder(images)
        image_features = image_outputs.logits  # now 2048-dim
        image_features = torch.flatten(image_features, start_dim=1)

        # Fusion
        combined = self.fusion_block(text=text_features, image=image_features)

        return combined


class CLIPDetectorWMemoryCoAttention(MemoryAugmentedDetector):
    def __init__(
        self,
        backbone,
        processor,
        out_dim: int = 2,
        use_memory: bool = True,
        memory_size: int = 512,
        memory_mode: str = "read_write",
        fusion_type: str = "concat",
        proj_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.5
    ):
        super().__init__(
            feature_dim=proj_dim * 2,
            out_dim=out_dim,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )
        self.backbone = backbone
        self.processor = processor

        # CLIP image_embeds/text_embeds are usually 512 each, but projectors keep this flexible
        clip_dim = getattr(backbone.config, "projection_dim", 512)

        self.image_proj = nn.Linear(clip_dim, proj_dim)
        self.text_proj = nn.Linear(clip_dim, proj_dim)

        self.co_attention = CoAttentionBlock(
            dim=proj_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)
        image_embeds = outputs.image_embeds   # [B, clip_dim]
        text_embeds = outputs.text_embeds     # [B, clip_dim]

        image_feats = self.image_proj(image_embeds).unsqueeze(1)  # [B, 1, D]
        text_feats = self.text_proj(text_embeds).unsqueeze(1)     # [B, 1, D]

        image_co, text_co = self.co_attention(image_feats, text_feats)

        fused = torch.cat(
            [image_co.squeeze(1), text_co.squeeze(1)],
            dim=1
        )  # [B, 2D]

        return fused


class NetShareFusionCLIP(nn.Module):
    def __init__(
        self,
            backbone,
            model_dim: int = 256,
            drop_and_BN: str = "drop-BN",
            num_labels: int = 2,
            num_layers: int = 2,
            num_heads: int = 8,
            ffn_dim: int = 2048,
            dropout: float = 0.5
    ):

        super().__init__()

        self.model_dim = model_dim
        self.drop_and_BN = drop_and_BN

        # CLIP text + image encoder
        self.clip = backbone
        clip_dim = self.clip.config.projection_dim  # usually 512

        self.linear_text = nn.Linear(clip_dim, model_dim)
        self.bn_text = nn.BatchNorm1d(model_dim)

        self.linear_image = nn.Linear(clip_dim, model_dim)
        self.bn_vgg = nn.BatchNorm1d(model_dim)

        self.dropout = nn.Dropout(dropout)

        # DCT image branch stays the same
        self.dct_img = DctCNN(
            model_dim,
            dropout,
            in_channel=128,
            branch1_channels=[64],
            branch2_channels=[48, 64],
            branch3_channels=[64, 96, 96],
            branch4_channels=[32],
            out_channels=64
        )

        self.linear_dct = nn.Linear(4096, model_dim)
        self.bn_dct = nn.BatchNorm1d(model_dim)

        # Multimodal fusion stays the same
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Classifier stays similar
        self.linear1 = nn.Linear(model_dim, 35)
        self.bn_1 = nn.BatchNorm1d(35)
        self.linear2 = nn.Linear(35, num_labels)

    def drop_BN_layer(self, x, part="dct"):
        if part == "dct":
            bn = self.bn_dct
        elif part == "vgg":
            bn = self.bn_vgg
        elif part == "bert":
            bn = self.bn_text

        if self.drop_and_BN == "drop-BN":
            x = self.dropout(x)
            x = bn(x)
        elif self.drop_and_BN == "BN-drop":
            x = bn(x)
            x = self.dropout(x)
        elif self.drop_and_BN == "drop-only":
            x = self.dropout(x)
        elif self.drop_and_BN == "BN-only":
            x = bn(x)
        elif self.drop_and_BN == "none":
            pass

        return x

    def forward(self, inputs, attn_mask=None):
        pixel_values = inputs['pixel_values']
        dct_img = inputs['dct_img']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # CLIP text + image features
        clip_outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        text_output = clip_outputs.text_embeds    # [B, clip_dim]
        image_output = clip_outputs.image_embeds  # [B, clip_dim]

        # Text feature
        text_output = F.relu(self.linear_text(text_output))
        text_output = self.drop_BN_layer(text_output, part="bert")

        # Image feature from CLIP image encoder
        image_output = F.relu(self.linear_image(image_output))
        image_output = self.drop_BN_layer(image_output, part="vgg")

        # DCT feature
        dct_out = self.dct_img(dct_img)
        dct_out = F.relu(self.linear_dct(dct_out))
        dct_out = self.drop_BN_layer(dct_out, part="dct")

        # Stage 1: CLIP image feature ↔ DCT feature
        output = image_output
        for fusion_layer in self.fusion_layers:
            output = fusion_layer(output, dct_out, attn_mask)

        # Stage 2: fused visual feature ↔ CLIP text feature
        for fusion_layer in self.fusion_layers:
            output = fusion_layer(output, text_output, attn_mask)

        # Classifier
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        logits = self.linear2(output)

        return logits
