import torch
import torch.nn as nn

from models.mem_llm import MemLLM


class MemoryTrainer:
    """
    A trainer for the memory-augmented LLM.
    It isolates the trainable parameters of the memory module and trains them
    using a language modeling objective.
    """

    def __init__(self, model: MemLLM, learning_rate: float = 1e-4):
        self.model = model

        # 1. Isolate and collect only the parameters that should be trained.
        # The LLM itself is frozen. We only train the memory bank and the injection layer.
        trainable_params = list(model.memory_bank.parameters()) + list(
            model.memory_injection.parameters()
        )

        print(f"Found {len(trainable_params)} trainable tensors.")
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        # Standard loss for language modeling. It ignores -100, which is useful for padding.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
        use_regularization: bool = True,
        sparsity_weight: float = 0.01,
        entropy_weight: float = 0.001,
    ) -> torch.Tensor:
        """
        Computes the total loss for training the memory module.
        """
        # 2.A. Primary Task Loss: Causal Language Modeling
        # To correctly calculate the loss for a Causal LM, we need to shift the
        # logits and labels so that the prediction for token i is compared to label i+1.
        augmented_logits = outputs["augmented_logits"]
        shift_logits = augmented_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        task_loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        total_loss = task_loss

        # 2.B. (Optional) Auxiliary Regularization Losses
        # These help the memory learn useful properties.
        if use_regularization and outputs["importance_scores"] is not None:
            importance_scores = outputs["importance_scores"]

            # Sparsity Loss (L1): Encourages the model to use fewer, more important memories.
            sparsity_loss = torch.mean(torch.abs(importance_scores))

            # Entropy Loss: Encourages the model to use a diverse set of memory slots
            # by penalizing distributions that are too sharp (low entropy).
            importance_dist = importance_scores / (
                torch.sum(importance_scores, dim=-1, keepdim=True) + 1e-10
            )
            entropy = -torch.sum(
                importance_dist * torch.log(importance_dist + 1e-10), dim=-1
            ).mean()
            entropy_loss = (
                -entropy
            )  # We want to maximize entropy, so we minimize its negative.

            total_loss += (
                sparsity_weight * sparsity_loss + entropy_weight * entropy_loss
            )

        return total_loss

    def train_step(self, batch: dict):
        """
        Performs a single training step (forward pass, loss calculation, backward pass, and optimization).
        """
        self.model.train()  # Set the model to training mode
        self.optimizer.zero_grad()

        # The batch should contain input_ids, attention_mask, and labels.
        outputs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        loss = self.compute_loss(outputs, batch["labels"])

        # Backpropagate the loss only to the trainable parameters
        loss.backward()

        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()
