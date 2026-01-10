from typing import Optional
import torch
import torch.nn as nn
from typing import Tuple


class LatentMemBank(nn.Module):
    def __init__(
        self,
        last_hidden_size: int,
        memory_dim: int = 512,
        num_memory_slots: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.num_memory_slots = num_memory_slots

        # 1. Memory encoder: Convert hidden state of llm into mem vector
        self.memory_encoder = nn.Sequential(
            nn.Linear(last_hidden_size, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim * 2, memory_dim),
        )

        # 2. Learnable memory slots
        self.memory_slots = nn.Parameter(
            torch.randn(num_memory_slots, memory_dim)  # [slots, mem_dim]
        )

        # 3. Cross attention: Using current mem as query to select relavant new experience
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 4. Update gate
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim), nn.Sigmoid()
        )

        # 5. Memory decoder: Convert memory vector back into LLM last hidden state
        self.memory_decoder = nn.Sequential(
            nn.Linear(memory_dim, last_hidden_size),
            nn.LayerNorm(last_hidden_size),
            nn.GELU(),
            nn.Linear(last_hidden_size, last_hidden_size),
        )

        # 6. Importance scoer: Jude the importance of memory
        self.importance_scorer = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode_experience(
        self, last_hidden_states: torch.Tensor  # [batch, seq_len, last_hidden_size]
    ) -> torch.Tensor:
        """Convert LLM's last hidden states into memory representation."""
        memory_repr = self.memory_encoder(
            last_hidden_states
        )  # [_,_,last_hidden_size -> mem_dim]
        return memory_repr

    def update_memory(
        self,
        current_memory: Optional[torch.Tensor],
        new_experience: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use cross attention to update memory.
        Return [updated memory, importance score]
        """
        batch_size = new_experience.shape[0]

        if current_memory is None:
            current_memory = self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)

        attended_memory, atten_weights = self.cross_attention(
            query=current_memory,
            key=new_experience,
            value=new_experience,
            key_padding_mask=mask,
        )

        # Gated update
        gate_input = torch.cat([current_memory, attended_memory], dim=-1)
        update_gate = self.update_gate(gate_input)

        updated_memory = (
            update_gate * attended_memory + (1 - update_gate) * current_memory
        )

        importance_scores = self.importance_scorer(updated_memory)

        return updated_memory, importance_scores.squeeze(-1)

    def retrieve_memory(
        self,
        memory: torch.Tensor,  # [batch, num_slots, memory_dim]
        query_hidden_states: torch.Tensor,  # [batch, seq_len, llm_hidden_size]
    ) -> torch.Tensor:
        """
        Search relavant memory and convert back to llm context.
        """
        query_memory = self.memory_encoder(query_hidden_states)

        # Calculate similarity. More similar, the dot product is larger.
        attn_scores = torch.bmm(query_memory, memory.transpose(1, 2)) / (
            self.memory_dim**0.5
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 检索记忆
        retrieved_memory = torch.bmm(
            attn_weights, memory
        )  # [batch, seq_len, memory_dim]

        # 解码回LLM空间
        memory_for_llm = self.memory_decoder(retrieved_memory)

        return memory_for_llm
