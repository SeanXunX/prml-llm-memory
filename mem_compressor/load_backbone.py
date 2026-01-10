import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
device = torch.device(device_name)
print(f"Using device: {device}")

# 1. Define your Custom Memory Module
class MyMemoryModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Example: A simple feed-forward network as a placeholder for your memory logic
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden_states):
        # Your memory logic goes here
        return self.network(hidden_states)


# 2. Define the Combined Model
class LLMWithMemory(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        # Load the backbone (Base model without the specific Language Model Head)
        # We use AutoModel instead of AutoModelForCausalLM because we want
        # the raw hidden states, not the vocabulary logits.
        self.backbone = AutoModel.from_pretrained(model_name)

        # FREEZE THE BACKBONE
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Initialize your memory module
        # We get the hidden size dynamically from the config (e.g., 768 for GPT-2 small)
        hidden_size = self.backbone.config.hidden_size
        self.memory_module = MyMemoryModule(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # 1. Pass input through the frozen LLM
        # output_hidden_states=True ensures we get the state of the last layer
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        # The 'last_hidden_state' has shape (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # 2. Pass the backbone output into your trainable memory module
        memory_output = self.memory_module(last_hidden_state)

        return memory_output


# --- Setup and Verification ---

# Initialize model with GPT-2
# For Llama, simply change the string (see section below)
model_name = "gpt2"
model = LLMWithMemory(model_name)

# Verify parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Backbone: {model_name}")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters (Memory Module only): {trainable_params:,}")

# Example Inference
input_ids = torch.randint(0, 50257, (1, 10))  # Fake inputs
output = model(input_ids)
print(f"Output shape: {output.shape}")
