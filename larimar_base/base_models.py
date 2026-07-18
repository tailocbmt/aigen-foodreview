import math
from typing import Tuple, Optional

from sklearn.neighbors import NearestNeighbors

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


class kNNMemory(nn.Module):
    """
    A non-parametric k-Nearest Neighbor memory baseline.
    Retrieves the top-k closest episodes using L2 distance and averages them.
    Contains zero learnable weights.
    """

    def __init__(
        self,
        memory_size: int,
        episode_dim: int,
        k: int = 50,  # The number of neighbors to retrieve
    ):
        super().__init__()
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.k = k

        # Raw feature buffers (No query_net, key_net, or value_net!)
        self.register_buffer("memory", torch.zeros(memory_size, episode_dim))
        self.register_buffer("memory_age", torch.zeros(memory_size))

    @torch.no_grad()
    def reset_memory(self):
        self.memory.zero_()
        self.memory_age.zero_()

    @torch.no_grad()
    def write_memory(self, episode: torch.Tensor):
        """Standard FIFO write policy, identical to the main model."""
        B = episode.size(0)

        if B > self.memory_size:
            episode = episode[:self.memory_size]
            B = self.memory_size

        _, idx = self.memory_age.topk(B, largest=False)

        new_memory = self.memory.clone()
        new_memory[idx] = episode.detach()
        self.memory = new_memory

        new_age = self.memory_age.max() + torch.arange(1, B + 1, device=episode.device)
        self.memory_age[idx] = new_age

    @torch.no_grad()
    def read_memory(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, D = query.shape

        # 1. Move data to CPU for sklearn (sklearn does not support GPU)
        memory_np = self.memory.cpu().numpy()  # shape: [memory_size, D]
        query_np = query.cpu().numpy()         # shape: [B, D]

        # 2. Build the sklearn KNN model and fit it to the memory
        # WARNING: Fitting inside read_memory rebuilds the tree EVERY inference step!
        nn = NearestNeighbors(n_neighbors=self.k,
                              metric='euclidean', algorithm='auto')
        nn.fit(memory_np)

        # 3. Get indices of the top-k smallest distances
        # distances shape: [B, k], topk_indices_np shape: [B, k]
        distances, topk_indices_np = nn.kneighbors(query_np)

        # 4. Move indices back to the original device (GPU if applicable)
        topk_indices = torch.tensor(topk_indices_np, device=query.device)

        # 5. Retrieve the actual feature vectors for those k slots (back on GPU)
        expanded_memory = self.memory.unsqueeze(0).expand(B, -1, -1)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        retrieved_k = torch.gather(expanded_memory, 1, topk_indices_expanded)

        # 6. Average the k neighbors
        retrieved = retrieved_k.max(dim=1)[0]

        # 7. Dummy attention matrix
        dummy_attn = torch.zeros(B, self.memory_size, device=query.device)
        dummy_attn.scatter_(1, topk_indices, 1.0 / self.k)

        return retrieved, dummy_attn

    def forward(
        self,
        episode: torch.Tensor,
        mode: str = "read_write",
        top_k: int = None,
        indexes=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mode == "write":
            return self.write_memory(episode), None
        elif mode == "read":
            retrieved, attention_weights = self.read_memory(episode)
            return retrieved, attention_weights, None
        else:
            retrieved, attention_weights = self.read_memory(episode)
            self.write_memory(episode)
            return retrieved, attention_weights, None


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
        mode: str = "read_write",
        indexes=None,
        top_k: int = None
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

        # 🔥 NEW: Buffer to track the generator ID (e.g., 0=Real, 1=SD, 2=Mistral)
        self.register_buffer("memory_labels", torch.zeros(
            memory_size, dtype=torch.long))

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

    def apply_rotary_emb_1d_old(self, xq, xk, freqs_cis):
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

    def apply_rotary_emb_1d(self, xq, xk, freqs_cis, query_pos, memory_pos):
        """
        xq: [B, D]
        xk: [M, D]
        freqs_cis: [max_len, D//2]
        query_pos: [B]
        memory_pos: [M]
        """
        # reshape into complex
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        # ✅ FIX: Gather the correct frequencies based on the actual temporal ages
        freqs_q = freqs_cis[query_pos]  # [B, D//2]
        freqs_k = freqs_cis[memory_pos]  # [M, D//2]

        # apply rotation using the properly mapped frequencies
        xq_out = torch.view_as_real(xq_ * freqs_q).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs_k).flatten(-2)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    # ---------------------
    # Memory ops
    # ---------------------

    @torch.no_grad()
    def reset_memory(self):
        self.memory.zero_()
        self.memory_age.zero_()
        self.memory_usage.zero_()

    # Remove this if you WANT gradients to flow into memory, keep it if you don't.
    @torch.no_grad()
    def write_memory(self, episode: torch.Tensor, indexes: torch.Tensor = None):
        B = episode.size(0)

        if B > self.memory_size:
            episode = episode[:self.memory_size]
            B = self.memory_size

        _, idx = self.memory_age.topk(B, largest=False)

        # ❌ OLD IN-PLACE: self.memory[idx] = episode.detach()

        # ✅ NEW OUT-OF-PLACE ASSIGNMENT:
        new_memory = self.memory.clone()
        new_memory[idx] = episode.detach()
        self.memory = new_memory  # Re-bind the buffer pointer safely

        # 🔥 NEW: Store the generator label alongside the features
        self.memory_labels[idx] = indexes.detach()

        # These are fine to do in-place because they don't require gradients
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
        top_k: int = None
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
        # q_rot, k_rot = self.apply_rotary_emb_1d(
        #     q, k,
        #     self.freqs_cis
        # )

        q_rot, k_rot = self.apply_rotary_emb_1d(
            q, k,
            self.freqs_cis,
            query_pos,
            memory_pos
        )

        # ---- attention ----
        attn_scores = torch.matmul(q_rot, k_rot.T) / math.sqrt(D)
        if top_k is None:
            attn = F.softmax(attn_scores, dim=-1)

            retrieved = torch.matmul(attn, v)
        else:
            # 2. Get the values and indices of the top K scores
            topk_scores, topk_indices = torch.topk(
                attn_scores, k=top_k, dim=-1)

            # 3. Create a mask of negative infinities
            sparse_scores = torch.full_like(attn_scores, float('-inf'))

            # 4. Scatter the top K scores back into the mask
            sparse_scores.scatter_(dim=-1, index=topk_indices, src=topk_scores)

            # 5. Apply Softmax. The -inf values become 0 weight.
            attn = F.softmax(sparse_scores, dim=-1)

            # 6. Retrieve (this is now a weighted average of ONLY the top 10)
            retrieved = torch.matmul(attn, v)

        with torch.no_grad():
            self.memory_usage += attn.sum(dim=0)

            # ✅ EXTRACTION: Calculate the exact distance between query and memory ages
            # query_pos is [B], memory_pos is [M]. We broadcast to get a [B, M] distance matrix.
            distance_matrix = torch.abs(
                query_pos.unsqueeze(1) - memory_pos.unsqueeze(0))

        return retrieved, attn, distance_matrix

    # ---------------------

    def forward(self, episode, mode="read_write", indexes=None, top_k=None):
        if mode == "write":
            self.write_memory(episode, indexes=indexes)
            return episode, None

        if mode == "read":
            return self.read_memory(episode, top_k=top_k)

        if mode == "read_write":
            out, attn, dist = self.read_memory(episode, top_k=top_k)
            self.write_memory(episode, indexes=indexes)
            return out, attn, dist

        raise ValueError(mode)


class MemoryAugmentedDetector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        out_dim: int = 1,
        use_memory: str = "linear",
        memory_size: int = 512,
        memory_mode: str = "read_write",
        fusion_type: str = "add",
        memory_architecture: str = "joint",  # 🔥 NEW: "joint" or "split"
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_memory = use_memory
        self.memory_mode = memory_mode
        self.fusion_type = fusion_type
        self.memory_architecture = memory_architecture

        self.feature_projection = None

        # 1. Initialize Memory/Memories based on architecture
        if use_memory in ["linear", "rope", "knn"]:
            MemoryClass = EpisodicMemory
            if use_memory == "linear":
                MemoryClass = EpisodicMemory
            elif use_memory == "rope":
                MemoryClass = EpisodicMemoryRoPE
            elif use_memory == "knn":
                MemoryClass = kNNMemory

            if self.memory_architecture == "split":
                # Assuming the joint feature_dim is evenly split between text and image (e.g., 512 + 512 = 1024)
                self.episodic_memory_t = MemoryClass(
                    memory_size=memory_size, episode_dim=312)
                self.episodic_memory_v = MemoryClass(
                    memory_size=memory_size, episode_dim=512)
            else:
                self.episodic_memory = MemoryClass(
                    memory_size=memory_size, episode_dim=feature_dim)
        else:
            self.use_memory = False

        # 2. Classifier dimensions
        classifier_in_dim = feature_dim * \
            2 if (self.use_memory and fusion_type == "concat") else feature_dim
        self.classifier = nn.Linear(classifier_in_dim, out_dim)

    @staticmethod
    def load_episodic_memory_weights_only(model, checkpoint_path: str):
        # 1. Load the saved weights (often nested under 'state_dict' or 'model')
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract the state dict if it's wrapped in a checkpoint dictionary
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint  # Assume the file is exactly the state_dict

        # 2. Filter the state_dict for ONLY the memory keys
        memory_state_dict = {}
        for key, value in state_dict.items():
            # Check for both joint and split prefixes
            if key.startswith("episodic_memory."):
                memory_state_dict[key] = value

        if not memory_state_dict:
            print("Warning: No episodic memory weights found in the checkpoint!")
            return model

        # 3. Load the filtered weights into the model
        # strict=False is REQUIRED because we are intentionally ignoring the rest of the model (e.g. feature_projection)
        model.load_state_dict(
            memory_state_dict, strict=False)

        print(
            f"Successfully loaded {len(memory_state_dict)} memory parameter tensors.")

        return model

    def fuse_with_memory(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([x, retrieved], dim=1)
        elif self.fusion_type == "add":
            return x + retrieved
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def apply_memory(self, x: torch.Tensor, indexes: torch.Tensor = None, return_memory: bool = False, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = {}
        if not self.use_memory or self.memory_mode == "off":
            return x, None

        if self.memory_architecture == "split":
            # 🔥 NEW: Chunk the joint representation into Text and Image halves
            x_t, x_v = x[0], x[1]

            if self.memory_mode in ["read", "read_write"]:
                # Read from respective memories
                retrieved_t, attn_t = self.episodic_memory_t(
                    x_t, self.memory_mode, indexes=indexes)
                retrieved_v, attn_v = self.episodic_memory_v(
                    x_v, self.memory_mode, indexes=indexes)

                # Fuse independently
                fused_t = self.fuse_with_memory(x_t, retrieved_t)
                fused_v = self.fuse_with_memory(x_v, retrieved_v)

                # Re-concatenate the fused halves into a single representation for the classifier
                x_out = torch.cat([fused_t, fused_v], dim=1)

                # Return a tuple of attention weights for analysis
                return x_out, (attn_t, attn_v)

            elif self.memory_mode == "write":
                return x, None

        else:
            # ORIGINAL LOGIC: Joint Memory
            if self.memory_mode in ["read", "read_write"]:
                retrieved, attention_weights, distance = self.episodic_memory(
                    x, self.memory_mode, indexes=indexes, top_k=top_k)
                x_out = self.fuse_with_memory(x, retrieved)

                if return_memory is True:
                    # ---------------------------------------------------------
                    # GET TOP 10 MOST RELEVANT MEMORIES
                    # k=10 gets the top 10, dim=-1 operates across the 512 memory slots
                    # ---------------------------------------------------------
                    topk_scores, topk_indices = torch.topk(
                        attention_weights, k=10, dim=-1)
                    # topk_scores: [B, 10] -> The actual attention weights of the top 10
                    # topk_indices: [B, 10] -> The row numbers (0 to 511) in self.memory

                    # If you want to extract the actual vectors of those top 10 slots:
                    # self.memory is shape [512, 824]. This will grab the exact rows.
                    # Shape: [B, 10, 824]
                    top10_vectors = self.episodic_memory.memory[topk_indices]

                    # Because you recently added self.memory_labels, you can also
                    # instantly check the generator IDs (e.g., Real, SD, Mistral) of those top 10:
                    # Shape: [B, 10]
                    top10_labels = self.episodic_memory.memory_labels[topk_indices]

                    outputs["x_input"] = x
                    outputs["x_output"] = x_out
                    outputs["retrieved_memory"] = retrieved
                    outputs["memory_attention_weights"] = attention_weights
                    outputs["distance"] = distance
                    outputs["topk_scores"] = topk_scores
                    outputs["topk_indices"] = topk_indices
                    outputs["top10_vectors"] = top10_vectors
                    outputs["top10_labels"] = top10_labels

                return x_out, outputs

            elif self.memory_mode == "write":
                return x, None

        raise ValueError(f"Invalid memory_mode: {self.memory_mode}")

    def set_memory_mode(self, mode: str):
        assert mode in ["read_write", "read", "write", "off"]
        self.memory_mode = mode

    def reset_memory(self):
        if self.use_memory:
            if self.memory_architecture == "split":
                if hasattr(self, "episodic_memory_t"):
                    self.episodic_memory_t.reset_memory()
                    self.episodic_memory_v.reset_memory()
            else:
                if hasattr(self, "episodic_memory"):
                    self.episodic_memory.reset_memory()

    def feature_extractor(self, inputs, return_interpretability: bool = False):
        raise NotImplementedError

    def forward(self, inputs, indexes=None, return_memory: bool = False, return_interpretability: bool = False, top_k: int = None):
        outputs = {}
        x, text_attention = self.feature_extractor(
            inputs, return_interpretability)

        if self.feature_projection is not None:
            x = self.feature_projection(x)

        x, outputs = self.apply_memory(
            x, indexes=indexes, return_memory=return_memory, top_k=top_k)

        logits = self.classifier(x)

        if return_memory:
            return logits, outputs
        if return_interpretability:
            return logits, text_attention

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
