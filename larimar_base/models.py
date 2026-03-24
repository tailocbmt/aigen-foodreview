import math
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BertModel,
    CLIPVisionModel,
)

logger = logging.getLogger(__name__)


# =========================================================
# ModelNet core layers
# =========================================================

class ModelNetLinear(nn.Module):
    """Simple linear layer replacement for easier training."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelNetAttention(nn.Module):
    """Multi-head attention using ModelNetLinear projections."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = ModelNetLinear(dim, dim, bias=bias)
        self.k_proj = ModelNetLinear(dim, dim, bias=bias)
        self.v_proj = ModelNetLinear(dim, dim, bias=bias)
        self.out_proj = ModelNetLinear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: [B, Q, D]
        key:   [B, K, D]
        value: [B, K, D]
        mask:  [B, K] or [B, Q, K]
        """
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, k_len, self.num_heads,
                   self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(
            q, k.transpose(-2, -1)) * self.scale  # [B, H, Q, K]

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, K]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, Q, K]
            mask = mask.expand(batch_size, self.num_heads, q_len, k_len)
            attention_scores = attention_scores.masked_fill(
                mask == 0, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, v)  # [B, H, Q, Hd]
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.dim)
        output = self.out_proj(attended)

        return output, attention_weights.mean(dim=1)  # [B, Q, K]


# =========================================================
# Pretrained encoders
# =========================================================

class PretrainedTextEncoder(nn.Module):
    """BERT text encoder wrapper."""

    def __init__(self, model_name: str = "bert-base-uncased", proj_dim: Optional[int] = None):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.proj = None
        if proj_dim is not None and proj_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, proj_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        token_features = outputs.last_hidden_state      # [B, T, D]
        pooled_features = outputs.pooler_output         # [B, D]

        if self.proj is not None:
            token_features = self.proj(token_features)
            pooled_features = self.proj(pooled_features)

        return {
            "token_features": token_features,
            "pooled_features": pooled_features
        }


class CLIPImageEncoder(nn.Module):
    """CLIP vision encoder wrapper."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", proj_dim: Optional[int] = None):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.proj = None
        if proj_dim is not None and proj_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, proj_dim)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            pixel_values=pixel_values,
            return_dict=True
        )

        token_features = outputs.last_hidden_state      # [B, V, D]
        pooled_features = outputs.pooler_output         # [B, D]

        if self.proj is not None:
            token_features = self.proj(token_features)
            pooled_features = self.proj(pooled_features)

        return {
            "token_features": token_features,
            "pooled_features": pooled_features
        }


# =========================================================
# Memory
# =========================================================

class EpisodicMemory(nn.Module):
    """Larimar-style episodic memory."""

    def __init__(
        self,
        memory_size: int,
        episode_dim: int,
        alpha: float = 0.1,
        direct_writing: bool = True
    ):
        super().__init__()
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.alpha = alpha
        self.direct_writing = direct_writing

        self.register_buffer("memory", torch.zeros(memory_size, episode_dim))
        self.register_buffer("memory_age", torch.zeros(memory_size))
        self.register_buffer("memory_usage", torch.zeros(memory_size))

        self.query_net = ModelNetLinear(episode_dim, episode_dim)
        self.key_net = ModelNetLinear(episode_dim, episode_dim)
        self.value_net = ModelNetLinear(episode_dim, episode_dim)

    def write_memory(self, episode: torch.Tensor) -> torch.Tensor:
        batch_size = episode.size(0)

        if self.direct_writing:
            # Fix: least recently used should use smallest age
            _, lru_indices = self.memory_age.topk(batch_size, largest=False)

            self.memory[lru_indices] = episode.detach()
            self.memory_age[lru_indices] = self.memory_age.max() + 1
            self.memory_usage[lru_indices] += 1

        return episode

    def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_net(query)              # [B, E]
        k = self.key_net(self.memory)          # [M, E]
        v = self.value_net(self.memory)        # [M, E]

        attention_scores = torch.matmul(
            q, k.transpose(0, 1)) / math.sqrt(self.episode_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, M]

        retrieved = torch.matmul(attention_weights, v)           # [B, E]
        self.memory_usage += attention_weights.sum(dim=0).detach()

        return retrieved, attention_weights

    def forward(self, episode: torch.Tensor, mode: str = "read_write") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mode == "write":
            return self.write_memory(episode), None
        elif mode == "read":
            return self.read_memory(episode)
        else:
            self.write_memory(episode)
            retrieved, attention_weights = self.read_memory(episode)
            return retrieved, attention_weights


# =========================================================
# Multimodal fusion
# =========================================================

class MultimodalCrossAttentionFusion(nn.Module):
    """
    Bidirectional token-level fusion:
    text attends to image, and image attends to text.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers

        self.text_to_image_attn = nn.ModuleList([
            ModelNetAttention(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.image_to_text_attn = nn.ModuleList([
            ModelNetAttention(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.text_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.image_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.output_proj = ModelNetLinear(hidden_dim, hidden_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        txt = text_tokens
        img = image_tokens
        attention_maps = {}

        for i in range(self.num_layers):
            txt_attended, txt2img_weights = self.text_to_image_attn[i](
                query=txt,
                key=img,
                value=img,
                mask=image_mask
            )
            txt = self.text_norms[i](txt + txt_attended)

            img_attended, img2txt_weights = self.image_to_text_attn[i](
                query=img,
                key=txt,
                value=txt,
                mask=text_mask
            )
            img = self.image_norms[i](img + img_attended)

            attention_maps[f"layer_{i}_text_to_image"] = txt2img_weights
            attention_maps[f"layer_{i}_image_to_text"] = img2txt_weights

        fused_text = self.output_proj(txt)
        fused_image = self.output_proj(img)

        return {
            "fused_text_tokens": fused_text,
            "fused_image_tokens": fused_image,
            "attention_maps": attention_maps
        }


# =========================================================
# Classifier model
# =========================================================

class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier:
    text + image -> fusion -> memory -> classifier -> real/fake
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        hidden_dim = config["fusion_hidden_size"]
        episode_dim = config["episode_dim"]

        # Encoders
        self.text_encoder = PretrainedTextEncoder(
            model_name=config.get("text_model_name", "bert-base-uncased"),
            proj_dim=hidden_dim
        )
        self.image_encoder = CLIPImageEncoder(
            model_name=config.get("vision_model_name",
                                  "openai/clip-vit-base-patch32"),
            proj_dim=hidden_dim
        )

        # Fusion
        self.fusion = MultimodalCrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=config["fusion_num_heads"],
            num_layers=config["fusion_num_layers"],
            dropout=config["dropout"]
        )

        # Memory
        self.memory = EpisodicMemory(
            memory_size=config["memory_size"],
            episode_dim=episode_dim,
            alpha=config.get("memory_alpha", 0.1),
            direct_writing=config.get("direct_writing", True)
        )

        # Episode projection
        self.text_to_episode = ModelNetLinear(hidden_dim, episode_dim)
        self.image_to_episode = ModelNetLinear(hidden_dim, episode_dim)

        # Classifier
        classifier_input_dim = hidden_dim + hidden_dim + episode_dim
        self.fc1 = nn.Linear(classifier_input_dim,
                             config["classifier_hidden_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        self.fc2 = nn.Linear(
            config["classifier_hidden_dim"], config["num_classes"])

        # Tokenizer handle
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("tokenizer_name", "bert-base-uncased")
        )

    @staticmethod
    def masked_mean_pool(token_tensor: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_tensor: [B, L, D]
        attention_mask: [B, L]
        """
        if attention_mask is None:
            return token_tensor.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()   # [B, L, 1]
        pooled = (token_tensor * mask).sum(dim=1) / \
            mask.sum(dim=1).clamp(min=1e-6)
        return pooled

    def create_episode(
        self,
        fused_text_tokens: torch.Tensor,
        fused_image_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        text_pooled = self.masked_mean_pool(
            fused_text_tokens, attention_mask)   # [B, H]
        image_pooled = fused_image_tokens.mean(
            dim=1)                            # [B, H]

        text_episode = self.text_to_episode(
            text_pooled)                         # [B, E]
        image_episode = self.image_to_episode(
            image_pooled)                      # [B, E]

        # [B, E]
        episode = text_episode + image_episode
        return episode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids:      [B, T]
        attention_mask: [B, T]
        pixel_values:   [B, C, H, W]
        labels:         [B] for classification
        """

        # 1) Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_tokens = text_outputs["token_features"]          # [B, T, H]

        # 2) Encode image
        image_outputs = self.image_encoder(pixel_values)
        image_tokens = image_outputs["token_features"]        # [B, V, H]

        # 3) Fuse
        fusion_outputs = self.fusion(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            text_mask=attention_mask,
            image_mask=None
        )
        fused_text_tokens = fusion_outputs["fused_text_tokens"]
        fused_image_tokens = fusion_outputs["fused_image_tokens"]

        # 4) Episode
        episode = self.create_episode(
            fused_text_tokens=fused_text_tokens,
            fused_image_tokens=fused_image_tokens,
            attention_mask=attention_mask
        )

        # 5) Memory
        if mode == "train":
            retrieved_memory, memory_attention = self.memory(
                episode, mode="read_write")
        else:
            retrieved_memory, memory_attention = self.memory(
                episode, mode="read")

        # 6) Pool final multimodal features
        text_pooled = self.masked_mean_pool(
            fused_text_tokens, attention_mask)   # [B, H]
        image_pooled = fused_image_tokens.mean(
            dim=1)                            # [B, H]

        # 7) Concatenate fused + memory
        final_repr = torch.cat(
            [text_pooled, image_pooled, retrieved_memory], dim=-1)

        # 8) Classifier
        x = self.fc1(final_repr)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)   # [B, num_classes]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        return {
            "loss": loss,
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "fused_text_tokens": fused_text_tokens,
            "fused_image_tokens": fused_image_tokens,
            "episode": episode,
            "retrieved_memory": retrieved_memory,
            "final_repr": final_repr,
            "fusion_attention": fusion_outputs["attention_maps"],
            "memory_attention": memory_attention,
            "memory_usage": self.memory.memory_usage.clone(),
        }


# =========================================================
# Helpers
# =========================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def create_multimodal_classifier(config: Dict) -> MultimodalClassifier:
    model = MultimodalClassifier(config)
    stats = count_parameters(model)
    logger.info(
        f"Model created with {stats['total_parameters']:,} total parameters")
    logger.info(f"Trainable parameters: {stats['trainable_parameters']:,}")
    return model


# =========================================================
# Example config
# =========================================================

example_config = {
    "text_model_name": "bert-base-uncased",
    "vision_model_name": "openai/clip-vit-base-patch32",
    "tokenizer_name": "bert-base-uncased",

    "fusion_hidden_size": 512,
    "fusion_num_heads": 8,
    "fusion_num_layers": 2,

    "memory_size": 1024,
    "episode_dim": 512,
    "memory_alpha": 0.1,
    "direct_writing": True,

    "classifier_hidden_dim": 256,
    "num_classes": 2,   # 0=fake, 1=real
    "dropout": 0.1,
}
