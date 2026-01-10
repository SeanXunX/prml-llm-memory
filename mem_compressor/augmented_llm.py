import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Tuple, Optional

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
device = torch.device(device_name)

# -----------------------------------------------------------------------------
# Memory Module (from mem_compressor/mem_compressor.py)
# -----------------------------------------------------------------------------


class DynamicLatentMemoryCompressor(nn.Module):
    """
    Dynamically compresses long-term history into a fixed-size latent memory bank.
    """

    def __init__(
        self,
        llm_hidden_size: int = 4096,
        memory_dim: int = 512,
        num_memory_slots: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.num_memory_slots = num_memory_slots

        self.memory_encoder = nn.Sequential(
            nn.Linear(llm_hidden_size, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim * 2, memory_dim),
        )

        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, memory_dim))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim), nn.Sigmoid()
        )

        self.memory_decoder = nn.Sequential(
            nn.Linear(memory_dim, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.importance_scorer = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode_experience(self, llm_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.memory_encoder(llm_hidden_states)

    def update_memory(
        self,
        current_memory: torch.Tensor,
        new_experience: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = new_experience.shape[0]

        if current_memory is None:
            current_memory = self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)

        attended_memory, _ = self.cross_attention(
            query=current_memory,
            key=new_experience,
            value=new_experience,
            key_padding_mask=mask,
        )

        gate_input = torch.cat([current_memory, attended_memory], dim=-1)
        update_gate = self.update_gate(gate_input)

        updated_memory = (
            update_gate * attended_memory + (1 - update_gate) * current_memory
        )

        importance_scores = self.importance_scorer(updated_memory)

        return updated_memory, importance_scores.squeeze(-1)

    def retrieve_memory(
        self, memory: torch.Tensor, query_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        query_memory = self.encode_experience(query_hidden_states)

        attn_scores = torch.bmm(query_memory, memory.transpose(1, 2)) / (
            self.memory_dim**0.5
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)

        retrieved_memory = torch.bmm(attn_weights, memory)

        return self.memory_decoder(retrieved_memory)


# -----------------------------------------------------------------------------
# Combined Model
# -----------------------------------------------------------------------------


class AugmentedLLM(nn.Module):
    """
    An LLM augmented with a dynamic, trainable memory bank.
    """

    def __init__(self, model_name: str, memory_config: dict = None):
        super().__init__()

        # Load the pretrained LLM with its language modeling head
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze the LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        hidden_size = self.llm.config.hidden_size

        # Initialize the trainable memory module
        memory_config = memory_config or {}
        self.memory_module = DynamicLatentMemoryCompressor(
            llm_hidden_size=hidden_size, **memory_config
        )

        # This buffer will hold the memory state across forward passes
        self.register_buffer("current_memory", None)

        # A trainable layer to inject memory information back into the LLM's processing
        self.memory_injection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        use_memory: bool = True,
    ):
        # 1. Get hidden states and logits from the frozen LLM
        with torch.no_grad():
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = llm_outputs.hidden_states[-1]

        # 2. Retrieve relevant memories and inject them
        if use_memory and self.current_memory is not None:
            retrieved_memory = self.memory_module.retrieve_memory(
                self.current_memory, hidden_states
            )
            combined = torch.cat([hidden_states, retrieved_memory], dim=-1)
            enhanced_hidden = self.memory_injection(combined)
        else:
            enhanced_hidden = hidden_states

        # 3. Update the memory bank with the new experience
        importance_scores = None
        if update_memory:
            new_experience = self.memory_module.encode_experience(hidden_states)

            # Create a mask to ignore padding tokens during memory update
            update_mask = (attention_mask == 0) if attention_mask is not None else None

            updated_memory, importance_scores = self.memory_module.update_memory(
                self.current_memory, new_experience, mask=update_mask
            )
            self.current_memory = updated_memory

        return {
            "enhanced_hidden_states": enhanced_hidden,
            "llm_logits": llm_outputs.logits,
            "memory_state": self.current_memory,
            "importance_scores": importance_scores,
        }

    def reset_memory(self):
        """Resets the memory state."""
        self.current_memory = None


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Using device: {device}")

    model_name = "gpt2"
    model = AugmentedLLM(model_name).to(device)

    # --- Verify Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nBackbone: {model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (Memory + Injection Layer): {trainable_params:,}")

    # --- Example Inference ---
    vocab_size = model.llm.config.vocab_size

    # 1. First pass: Memory is empty, so it only gets updated.
    print("\n--- Pass 1: Writing to memory ---")
    input_ids_1 = torch.randint(0, vocab_size, (1, 10), device=device)
    outputs_1 = model(input_ids_1)
    print(f"Output hidden state shape: {outputs_1['enhanced_hidden_states'].shape}")
    print(f"Logits shape: {outputs_1['llm_logits'].shape}")
    if outputs_1["memory_state"] is not None:
        print(
            f"Memory state is now populated. Shape: {outputs_1['memory_state'].shape}"
        )

    # 2. Second pass: Memory is read from, used, and then updated again.
    print("\n--- Pass 2: Reading from and updating memory ---")
    input_ids_2 = torch.randint(0, vocab_size, (1, 15), device=device)
    outputs_2 = model(input_ids_2)
    print(f"Output hidden state shape: {outputs_2['enhanced_hidden_states'].shape}")
    if outputs_2["memory_state"] is not None:
        print(f"Memory state updated. Shape: {outputs_2['memory_state'].shape}")

    # 3. Reset memory for a clean state.
    model.reset_memory()
    print("\n--- Memory Reset ---")
    print("Memory state after reset:", model.current_memory)

    # 4. Pass after reset: Behaves like the first pass again.
    print("\n--- Pass 3: After reset ---")
    outputs_3 = model(input_ids_1)
    print(f"Memory state is populated again. Shape: {outputs_3['memory_state'].shape}")
