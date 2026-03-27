import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """Simple episodic memory bank for one modality."""

    def __init__(
        self,
        memory_size: int,
        episode_dim: int,
        direct_writing: bool = True,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.episode_dim = episode_dim
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
        """
        episode: [B, D]
        """
        batch_size = episode.size(0)

        if batch_size > self.memory_size:
            episode = episode[:self.memory_size]
            batch_size = self.memory_size

        # true-ish age update: older entries get larger age
        self.memory_age += 1

        # evict oldest
        _, lru_indices = self.memory_age.topk(batch_size, largest=True)

        self.memory[lru_indices] = episode.detach()
        self.memory_age[lru_indices] = 0
        self.memory_usage[lru_indices] += 1

        return episode

    def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: [B, D]
        returns:
            retrieved: [B, D]
            attention_weights: [B, M]
        """
        q = self.query_net(query)         # [B, D]
        k = self.key_net(self.memory)     # [M, D]
        v = self.value_net(self.memory)   # [M, D]

        attention_scores = torch.matmul(
            q, k.transpose(0, 1)) / math.sqrt(self.episode_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, M]
        retrieved = torch.matmul(attention_weights, v)           # [B, D]

        with torch.no_grad():
            self.memory_usage += attention_weights.sum(dim=0).detach()

        return retrieved, attention_weights


class DualMemoryFusion(nn.Module):
    """
    Fuses current and retrieved features for text and image separately,
    then merges both modalities.
    """

    def __init__(self, embed_dim: int, out_dim: int = 1, fusion_type: str = "concat"):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            # [img, txt, retrieved_img, retrieved_txt] = 4 * D
            classifier_in_dim = embed_dim * 4
        elif fusion_type == "gated_add":
            # gated fusion returns one D for image and one D for text => 2 * D
            classifier_in_dim = embed_dim * 2
            self.img_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
            self.txt_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = nn.Linear(classifier_in_dim, out_dim)

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        retrieved_image: torch.Tensor,
        retrieved_text: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion_type == "concat":
            fused = torch.cat(
                [image_embeds, text_embeds, retrieved_image, retrieved_text], dim=1
            )
        else:
            img_gate = self.img_gate(
                torch.cat([image_embeds, retrieved_image], dim=1))
            txt_gate = self.txt_gate(
                torch.cat([text_embeds, retrieved_text], dim=1))

            fused_image = img_gate * retrieved_image + \
                (1.0 - img_gate) * image_embeds
            fused_text = txt_gate * retrieved_text + \
                (1.0 - txt_gate) * text_embeds
            fused = torch.cat([fused_image, fused_text], dim=1)

        logits = self.classifier(fused)
        return logits


class CLIPDetectorSeparateMemory(nn.Module):
    """
    CLIP detector with separate image memory and text memory.

    Important:
    - forward only READS memory
    - write_memory(...) should be called manually after optimizer.step()
      to avoid in-place autograd issues
    """

    def __init__(
        self,
        backbone,
        processor,
        embed_dim: int = 512,
        out_dim: int = 1,
        use_memory: bool = True,
        memory_size: int = 512,
        memory_mode: str = "read",
        fusion_type: str = "concat",   # "concat" or "gated_add"
    ):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.embed_dim = embed_dim
        self.use_memory = use_memory
        self.memory_mode = memory_mode

        if use_memory:
            self.image_memory = EpisodicMemory(
                memory_size=memory_size,
                episode_dim=embed_dim
            )
            self.text_memory = EpisodicMemory(
                memory_size=memory_size,
                episode_dim=embed_dim
            )
        else:
            self.image_memory = None
            self.text_memory = None

        self.fusion_head = DualMemoryFusion(
            embed_dim=embed_dim,
            out_dim=out_dim,
            fusion_type=fusion_type
        )

    def reset_memory(self):
        if self.image_memory is not None:
            self.image_memory.reset_memory()
        if self.text_memory is not None:
            self.text_memory.reset_memory()

    def feature_extractor(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(**inputs)
        # [B, 512] for clip-vit-base-patch16
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds     # [B, 512]
        return text_embeds, image_embeds

    def apply_memory(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        returns:
            retrieved_text, retrieved_image, text_attention, image_attention
        """
        if (not self.use_memory) or self.memory_mode == "off":
            return text_embeds, image_embeds, None, None

        if self.memory_mode in ["read", "read_write"]:
            retrieved_text, text_attention = self.text_memory.read_memory(
                text_embeds)
            retrieved_image, image_attention = self.image_memory.read_memory(
                image_embeds)
            return retrieved_text, retrieved_image, text_attention, image_attention

        if self.memory_mode == "write":
            return text_embeds, image_embeds, None, None

        raise ValueError(f"Invalid memory_mode: {self.memory_mode}")

    @torch.no_grad()
    def write_memory(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        """
        Call this AFTER optimizer.step() during training.
        """
        if not self.use_memory:
            return

        self.text_memory.write_memory(text_embeds)
        self.image_memory.write_memory(image_embeds)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_attention: bool = False,
        return_features: bool = False,
    ):
        text_embeds, image_embeds = self.feature_extractor(inputs)

        retrieved_text, retrieved_image, text_attention, image_attention = self.apply_memory(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )

        logits = self.fusion_head(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            retrieved_image=retrieved_image,
            retrieved_text=retrieved_text,
        )

        outputs = [logits]

        if return_attention:
            outputs.append({
                "text_attention": text_attention,
                "image_attention": image_attention,
            })

        if return_features:
            outputs.append({
                "text_embeds": text_embeds.detach(),
                "image_embeds": image_embeds.detach(),
            })

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
