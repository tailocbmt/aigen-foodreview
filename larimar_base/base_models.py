import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDetector(nn.Module):
    def __init__(self):
        super().__init__()

    def set_memory_mode(self, mode: str = ''):
        pass

    def reset_memory(self):
        pass


class CLIPDetector(BaseDetector):
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


class FLAVADetector(BaseDetector):
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

# =========================
# Episodic Memory with RoPE
# =========================


class EpisodicMemoryRoPE(nn.Module):
    """
    Larimar-style episodic memory with RoPE applied to queries and keys.

    RoPE can use either:
    - memory_age: temporal write order, recommended
    - slot: fixed memory slot index
    """

    def __init__(
        self,
        memory_size: int,
        episode_dim: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        assert episode_dim % 2 == 0, "RoPE requires even dimension"

        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.rope_theta = rope_theta

        self.register_buffer("memory", torch.zeros(memory_size, episode_dim))
        self.register_buffer("memory_age", torch.zeros(memory_size))
        self.register_buffer("memory_usage", torch.zeros(memory_size))

        self.query_net = nn.Linear(episode_dim, episode_dim)
        self.key_net = nn.Linear(episode_dim, episode_dim)
        self.value_net = nn.Linear(episode_dim, episode_dim)

        # Precompute large enough RoPE table
        self.max_positions = 10000
        self.register_buffer(
            "freqs_cis",
            self.compute_cis_1d(episode_dim, self.max_positions, rope_theta),
        )

    def compute_cis_1d(self, dim: int, seq_len: int, theta: float = 10000.0):
        """
        1D RoPE frequencies (complex exponential form)
        returns: [seq_len, dim//2] complex tensor
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(seq_len).float()

        freqs = torch.outer(positions, freqs)  # [seq_len, dim/2]

        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex
        return freqs_cis  # [seq_len, dim/2]

    def apply_rotary_emb_1d(self, xq, xk, freqs_cis):
        """
        xq: [B, D]
        xk: [M, D]
        freqs_cis: [max_len, D//2]
        """
        # reshape into complex
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        # apply rotation
        xq_out = torch.view_as_real(xq_ * freqs_cis[:xq_.shape[0]]).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs_cis[:xk_.shape[0]]).flatten(-2)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    # ---------------------
    # Memory ops
    # ---------------------

    @torch.no_grad()
    def reset_memory(self):
        self.memory.zero_()
        self.memory_age.zero_()
        self.memory_usage.zero_()

    @torch.no_grad()
    def write_memory(self, episode: torch.Tensor):
        B = episode.size(0)

        if B > self.memory_size:
            episode = episode[:self.memory_size]
            B = self.memory_size

        _, idx = self.memory_age.topk(B, largest=False)

        self.memory[idx] = episode.detach()

        new_age = self.memory_age.max() + torch.arange(
            1, B + 1, device=episode.device
        )
        self.memory_age[idx] = new_age
        self.memory_usage[idx] += 1

    # ---------------------
    # RoPE-aware read
    # ---------------------

    def read_memory(
        self,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, D = query.shape

        q = self.query_net(query)      # [B, D]
        k = self.key_net(self.memory)  # [M, D]
        v = self.value_net(self.memory)

        # ---- positions ----
        memory_pos = self.memory_age.long().clamp(max=self.max_positions - 1)
        query_pos = (
            self.memory_age.max().long() + 1 +
            torch.arange(B, device=query.device)
        ).clamp(max=self.max_positions - 1)

        # ---- apply RoPE ----
        q_rot, k_rot = self.apply_rotary_emb_1d(
            q, k,
            self.freqs_cis
        )

        # ---- attention ----
        attn_scores = torch.matmul(q_rot, k_rot.T) / math.sqrt(D)
        attn = F.softmax(attn_scores, dim=-1)

        retrieved = torch.matmul(attn, v)

        with torch.no_grad():
            self.memory_usage += attn.sum(dim=0)

        return retrieved, attn

    # ---------------------

    def forward(self, episode, mode="read_write"):
        if mode == "write":
            self.write_memory(episode)
            return episode, None

        if mode == "read":
            return self.read_memory(episode)

        if mode == "read_write":
            out, attn = self.read_memory(episode)
            self.write_memory(episode)
            return out, attn

        raise ValueError(mode)


class MemoryAugmentedDetector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        out_dim: int = 1,
        use_memory: str = "linear",
        memory_size: int = 512,
        memory_mode: str = "read_write",
        fusion_type: str = "concat"
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_memory = use_memory
        self.memory_mode = memory_mode
        self.fusion_type = fusion_type

        if use_memory == "linear":
            self.episodic_memory = EpisodicMemory(
                memory_size=memory_size,
                episode_dim=feature_dim
            )
        elif use_memory == "rope":
            self.episodic_memory = EpisodicMemoryRoPE(
                memory_size=memory_size,
                episode_dim=feature_dim
            )
        else:
            self.episodic_memory = None

        classifier_in_dim = feature_dim * \
            2 if (use_memory and fusion_type == "concat") else feature_dim
        self.classifier = nn.Linear(classifier_in_dim, out_dim)

    def fuse_with_memory(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([x, retrieved], dim=1)
        elif self.fusion_type == "add":
            return x + retrieved
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def apply_memory(self, x: torch.Tensor):
        attention_weights = None

        if not self.use_memory or self.memory_mode == "off":
            return x, attention_weights

        if self.memory_mode == "read":
            retrieved, attention_weights = self.episodic_memory(
                x, self.memory_mode)
            x = self.fuse_with_memory(x, retrieved)
            return x, attention_weights

        elif self.memory_mode == "read_write":
            # only READ here
            retrieved, attention_weights = self.episodic_memory(
                x, self.memory_mode)
            x = self.fuse_with_memory(x, retrieved)
            return x, attention_weights

        elif self.memory_mode == "write":
            return x, attention_weights

        else:
            raise ValueError(f"Invalid memory_mode: {self.memory_mode}")

    # 🔥 NEW: clean control functions
    def set_memory_mode(self, mode: str):
        assert mode in ["read_write", "read", "off"]
        self.memory_mode = mode

    def reset_memory(self):
        if self.use_memory and getattr(self, "episodic_memory", None) is not None:
            self.episodic_memory.reset_memory()

    def feature_extractor(self, inputs):
        raise NotImplementedError

    def forward(self, inputs, return_attention: bool = False):
        x = self.feature_extractor(inputs)
        x, attention_weights = self.apply_memory(x)
        logits = self.classifier(x)

        if return_attention:
            return logits, attention_weights
        return logits


class CLIPDetectorWMemory(MemoryAugmentedDetector):
    def __init__(
        self,
        backbone,
        processor,
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
        self.backbone = backbone
        self.processor = processor

    def feature_extractor(self, inputs):
        outputs = self.backbone(**inputs)
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds
        fused = torch.cat([image_embeds, text_embeds], dim=1)

        return fused


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
        embeddings = outputs.multimodal_embeddings
        cls_embedding = embeddings[:, 0, :]
        return cls_embedding
