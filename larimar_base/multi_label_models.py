import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import ResNetForImageClassification, BertForSequenceClassification
from larimar_base.base_models import BaseDetector, MemoryAugmentedDetector
from larimar_base.utils import DctCNN, PositionalWiseFeedForward, multimodal_fusion_layer


class FakeNewsSeparate(BaseDetector):
    def __init__(self, text_weights_dir: str = "", image_weights_dir: str = "", output_dim: int = 2):
        super().__init__()

        # Text encoder
        self.text_encoder = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Replace with the base model you used
            ignore_mismatched_sizes=True
        )

        # Image encoder (Hugging Face)
        self.image_encoder = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50", num_labels=output_dim, ignore_mismatched_sizes=True)

        # 4. Inject your dynamically found latest weights
        if text_weights_dir != "":
            text_state_dict = torch.load(text_weights_dir)
            self.text_encoder.load_state_dict(text_state_dict)
        if image_weights_dir != "":
            image_state_dict = torch.load(image_weights_dir)
            self.image_encoder.load_state_dict(image_state_dict)

    def forward(self, inputs):
        images = inputs['pixel_values']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # Text
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)

        # Image
        image_outputs = self.image_encoder(images)

        # Extract the logits from the model outputs
        text_val = torch.softmax(text_outputs.logits, dim=-1)
        # 1. Remove .tolist() and add .unsqueeze(1) to make shape [B, 1]
        text_pred = torch.argmax(text_val, dim=-1).unsqueeze(1)

        image_val = torch.softmax(image_outputs.logits, dim=-1)
        # 2. Remove .tolist() and add .unsqueeze(1) to make shape [B, 1]
        image_pred = torch.argmax(image_val, dim=-1).unsqueeze(1)

        # 3. Now you can safely concatenate them into a [B, 2] tensor
        logits = torch.cat((text_pred, image_pred), dim=1)

        return logits


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
        # resnet50 + bert: 2816; resnet18 + tinybert: 824; resnet18 + distilbert: 1280
        feature_dim=1280,
        memory_mode="read_write",
        fusion_type="add",
        text_backbone="bert-base-uncased",
        vision_backbone="microsoft/resnet-50"
    ):
        super().__init__(
            feature_dim=feature_dim,
            out_dim=out_dim,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )
        image_dim = 2048  # resnet50 + bert: 2048; resnet18 + bert: 512

        self.text_encoder = BertModel.from_pretrained(
            text_backbone, attn_implementation="eager")
        text_dim = self.text_encoder.config.hidden_size  # 768

        # Image encoder (Hugging Face)
        self.image_encoder = ResNetForImageClassification.from_pretrained(
            vision_backbone,
            num_labels=out_dim,
            ignore_mismatched_sizes=True
        )

        # Remove classification head → get features
        self.image_encoder.classifier = nn.Identity()

        # Memory dim
        memory_dim = text_dim + image_dim

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def feature_extractor(self, inputs, return_interpretability: bool = False):
        images = inputs['pixel_values']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # Text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_interpretability  # Expose attentions
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Extract text attention if requested
        text_attention = None
        if return_interpretability:
            # attentions is a tuple of layers. We want the last layer: shape (B, Num_Heads, Seq_Len, Seq_Len)
            last_layer_attn = text_outputs.attentions[-1]
            # Average across all heads for the [CLS] token (index 0)
            text_attention = last_layer_attn[:, :, 0, :].mean(dim=1)

        # Image
        image_outputs = self.image_encoder(images)
        image_features = image_outputs.logits  # now 2048-dim
        image_features = torch.flatten(image_features, start_dim=1)

        # Fusion
        combined = torch.cat((text_features, image_features), dim=1)

        return combined, text_attention


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


class FakeNewsMultimodalWMemoryCoAttention(MemoryAugmentedDetector):
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
            fusion_dim: int = 256):

        super().__init__(
            feature_dim=proj_dim * 2,
            out_dim=out_dim,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )
        # Text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_dim = self.text_encoder.config.hidden_size  # 768

        # Image encoder (Hugging Face)
        self.image_encoder = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=out_dim,
            ignore_mismatched_sizes=True
        )

        # Remove classification head → get features
        self.image_encoder.classifier = nn.Identity()
        self.image_dim = 2048

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
            nn.Linear(fusion_dim * 4, 512),
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


class NetShareFusionCLIP(MemoryAugmentedDetector):
    def __init__(
        self,
        use_memory: str = "linear",
        memory_size: int = 512,
        memory_mode: str = "read_write",
        fusion_type: str = "add",
        model_dim: int = 256,
        num_labels: int = 2,
        dropout: float = 0.5
    ):

        super().__init__(
            feature_dim=3072,
            out_dim=num_labels,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )

        self.model_dim = model_dim

        # CLIP text + image encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size  # 768

        # Image encoder (Hugging Face)
        self.image_encoder = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Remove classification head → get features
        self.image_encoder.classifier = nn.Identity()
        image_dim = 2048

        # DCT image branch stays the same
        self.dct_img = DctCNN(
            self.model_dim,
            dropout,
            in_channel=128,
            branch1_channels=[64],
            branch2_channels=[48, 64],
            branch3_channels=[64, 96, 96],
            branch4_channels=[32],
            out_channels=64
        )

        self.linear_dct = nn.Sequential(
            nn.Linear(4096, self.model_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        memory_dim = text_dim + image_dim + self.model_dim
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def feature_extractor(self, inputs):
        pixel_values = inputs['pixel_values']
        dct_img = inputs['dct_img']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        text_output = text_outputs.last_hidden_state[:, 0, :]

        # Image
        image_outputs = self.image_encoder(pixel_values)
        image_features = image_outputs.logits  # now 2048-dim
        image_output = torch.flatten(image_features, start_dim=1)

        # DCT feature
        dct_out = self.dct_img(dct_img)
        # 2. Reshape back to (Batch, Channels, Height, Width)
        # Assuming out_channels=64 and a square spatial dimension (64x64)
        dct_out = dct_out.view(-1, 64, 64, 64)
        # 3. Pool down to an 8x8 spatial size
        dct_out = F.adaptive_avg_pool2d(dct_out, (8, 8))
        dct_out = torch.flatten(dct_out, start_dim=1)
        dct_out = self.linear_dct(dct_out)

        combined = torch.cat((text_output, image_output, dct_out), dim=1)

        return combined
