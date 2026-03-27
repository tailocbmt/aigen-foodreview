import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPDetector(nn.Module):
    def __init__(self, backbone, processor, out_dim=1):
        super(CLIPDetector, self).__init__()
        self.backbone = backbone
        self.processor = processor
        self.fc1 = nn.Linear(1024, out_dim)

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds
        return text_embeds, image_embeds

    def forward(self, inputs):
        text_embeds, image_embeds = self.feature_extractor(inputs)
        # this fusion is just a simple concatenation
        fused = torch.cat([image_embeds, text_embeds], dim=1)
        output = self.fc1(fused)
        return output


class FLAVADetector(nn.Module):
    def __init__(self, backbone, processor, out_dim=1):
        super(FLAVADetector, self).__init__()
        self.backbone = backbone
        self.processor = processor
        self.fc1 = nn.Linear(768, out_dim)

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)
        embeddings = outputs.multimodal_embeddings
        cls_embedding = embeddings[:, 0, :]
        return cls_embedding

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        return self.fc1(x)


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

        self.query_net = nn.Linear(episode_dim, episode_dim)
        self.key_net = nn.Linear(episode_dim, episode_dim)
        self.value_net = nn.Linear(episode_dim, episode_dim)

    @torch.no_grad()
    def reset_memory(self):
        self.memory.zero_()
        self.memory_age.zero_()
        self.memory_usage.zero_()

    @torch.no_grad()
    def write_memory(self, episode: torch.Tensor) -> torch.Tensor:
        batch_size = episode.size(0)

        if batch_size > self.memory_size:
            episode = episode[:self.memory_size]
            batch_size = self.memory_size

        _, lru_indices = self.memory_age.topk(batch_size, largest=False)

        self.memory[lru_indices] = episode.detach()
        self.memory_age[lru_indices] = self.memory_age.max() + 1
        self.memory_usage[lru_indices] += 1

        return episode

    def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: [B, D]
        returns:
            retrieved: [B, D]
            attention_weights: [B, M]
        """
        q = self.query_net(query)         # [B, E]
        k = self.key_net(self.memory)     # [M, E]
        v = self.value_net(self.memory)   # [M, E]

        attention_scores = torch.matmul(
            q, k.transpose(0, 1)) / math.sqrt(self.episode_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        retrieved = torch.matmul(attention_weights, v)

        with torch.no_grad():
            self.memory_usage += attention_weights.sum(dim=0).detach()

        return retrieved, attention_weights

    def forward(
        self,
        episode: torch.Tensor,
        mode: str = "read_write"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mode == "write":
            return self.write_memory(episode), None
        elif mode == "read":
            return self.read_memory(episode)
        else:
            retrieved, attention_weights = self.read_memory(episode)
            self.write_memory(episode)
            return retrieved, attention_weights


class MemoryAugmentedDetector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        out_dim: int = 1,
        use_memory: bool = True,
        memory_size: int = 512,
        memory_mode: str = "read_write",
        fusion_type: str = "concat"
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_memory = use_memory
        self.memory_mode = memory_mode
        self.fusion_type = fusion_type

        if use_memory:
            self.episodic_memory = EpisodicMemory(
                memory_size=memory_size,
                episode_dim=feature_dim
            )
        else:
            self.episodic_memory = None

        if fusion_type == "concat":
            classifier_in_dim = feature_dim * 2
            self.gate_net = None

        elif fusion_type == "add":
            classifier_in_dim = feature_dim
            self.gate_net = None

        elif fusion_type == "gated_add":
            classifier_in_dim = feature_dim
            self.gate_net = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = nn.Linear(classifier_in_dim, out_dim)

    def fuse_with_memory(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([x, retrieved], dim=1)

        elif self.fusion_type == "add":
            return x + retrieved

        elif self.fusion_type == "gated_add":
            gate_input = torch.cat([x, retrieved], dim=1)   # [B, 2D]
            gate = self.gate_net(gate_input)                # [B, D]
            return gate * retrieved + (1.0 - gate) * x
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def apply_memory(self, x: torch.Tensor):
        attention_weights = None

        if not self.use_memory or self.memory_mode == "off":
            return x, attention_weights

        if self.memory_mode == "read":
            retrieved, attention_weights = self.episodic_memory.read_memory(x)
            x = self.fuse_with_memory(x, retrieved)
            return x, attention_weights

        elif self.memory_mode == "read_write":
            # only READ here
            retrieved, attention_weights = self.episodic_memory.read_memory(x)
            x = self.fuse_with_memory(x, retrieved)
            return x, attention_weights

        elif self.memory_mode == "write":
            return x, attention_weights

        else:
            raise ValueError(f"Invalid memory_mode: {self.memory_mode}")

    def feature_extractor(self, inputs):
        raise NotImplementedError

    def forward(
        self,
        inputs,
        return_attention: bool = False,
        return_features: bool = False
    ):
        raw_x = self.feature_extractor(inputs)   # [B, D]
        x, attention_weights = self.apply_memory(raw_x)
        logits = self.classifier(x)

        outputs = [logits]

        if return_attention:
            outputs.append(attention_weights)

        if return_features:
            outputs.append(raw_x.detach())   # write this to memory later

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class CLIPDetectorWMemory(MemoryAugmentedDetector):
    def __init__(
        self,
        backbone,
        processor,
        out_dim=1,
        use_memory=True,
        memory_size=512,
        memory_mode="read_write",
        fusion_type="concat"
    ):
        super().__init__(
            feature_dim=1024,  # 1024
            out_dim=out_dim,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )
        self.backbone = backbone
        self.processor = processor

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        x = torch.cat([image_embeds, text_embeds], dim=1)

        return x


class FLAVADetectorWMemory(MemoryAugmentedDetector):
    def __init__(
        self,
        backbone,
        processor,
        out_dim=1,
        use_memory=True,
        memory_size=512,
        memory_mode="read_write",
        fusion_type="concat"
    ):
        super().__init__(
            feature_dim=768,
            out_dim=out_dim,
            use_memory=use_memory,
            memory_size=memory_size,
            memory_mode=memory_mode,
            fusion_type=fusion_type
        )
        self.backbone = backbone
        self.processor = processor

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)

        # embeddings = outputs.multimodal_embeddings
        # cls_embedding = embeddings[:, 0, :]

        # unimodal branches
        text_embeddings = outputs.text_embeddings      # [B, T_text, 768]
        image_embeddings = outputs.image_embeddings    # [B, T_img, 768]

        # usually first token is CLS
        text_cls = text_embeddings[:, 0, :]            # [B, 768]
        image_cls = image_embeddings[:, 0, :]          # [B, 768]

        diff = torch.abs(image_cls - text_cls)
        prod = image_cls * text_cls

        fused = torch.cat([image_cls, text_cls, diff, prod], dim=1)

        return fused
