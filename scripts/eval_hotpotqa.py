from typing import Literal

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from models.mem_llm import MemLLM
from utils.data_loader import get_hotpot_dataloader
from utils.metrics import calculate_metrics


@torch.no_grad()
def generate_answers(
    model: MemLLM,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    device: str,
    max_new_tokens: int = 50,
):
    """
    Generates answers for the entire validation set.
    """
    model.eval()
    predictions = []
    ground_truths = []

    for batch in tqdm(dataloader, desc="Generating Answers"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # For generation, we only need the prompt part of the input
        # We find the length of the prompt by finding the start of the labels
        # (where labels are not -100)
        prompt_end_index = (batch["labels"][0] != -100).nonzero(as_tuple=True)[0][0]
        prompt_input_ids = input_ids[:, :prompt_end_index]
        prompt_attention_mask = attention_mask[:, :prompt_end_index]

        # --- Autoregressive Generation Loop ---
        model.current_memory = None  # Reset memory for each new prompt
        generated_ids = prompt_input_ids

        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=generated_ids, attention_mask=prompt_attention_mask
            )
            # Get the logits for the very last token
            next_token_logits = outputs["augmented_logits"][:, -1, :]
            # Use greedy decoding to get the most likely next token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            prompt_attention_mask = torch.cat(
                [prompt_attention_mask, torch.ones_like(next_token)], dim=1
            )

            # Stop if the model generates an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode the generated answer and the ground truth
        # The generated answer is the part after the initial prompt
        generated_answer = tokenizer.decode(
            generated_ids[0][prompt_input_ids.shape[1] :], skip_special_tokens=True
        )
        # The ground truth is the part of the original input that was masked
        ground_truth_ids = batch["input_ids"][0][prompt_input_ids.shape[1] :]
        ground_truth = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)

        predictions.append(generated_answer)
        ground_truths.append(ground_truth)

    return predictions, ground_truths


def get_deivce() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    # --- Configuration ---
    MODEL_NAME = "gpt2"
    CHECKPOINT_PATH = "hotpotqa_memory_module_final.pt"
    BATCH_SIZE = 1  # Evaluate one by one for generation
    DEVICE = get_deivce()

    logger.add("logs/evaluation.log")
    logger.info(f"Starting evaluation on device: {DEVICE}")

    # --- Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model and fine-tuned memory weights...")
    model = MemLLM(MODEL_NAME).to(DEVICE)
    # Load only the trainable memory module weights
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False
    )

    # --- Load Data ---
    logger.info("Loading validation data...")
    val_dataloader = get_hotpot_dataloader(
        "validation", tokenizer, batch_size=BATCH_SIZE
    )

    # --- Generate and Evaluate ---
    predictions, ground_truths = generate_answers(
        model, val_dataloader, tokenizer, DEVICE
    )

    logger.info("Calculating metrics...")
    metrics = calculate_metrics(predictions, ground_truths)

    logger.success("Evaluation finished!")
    logger.success(f"Exact Match: {metrics['exact_match']:.2f}%")
    logger.success(f"F1 Score: {metrics['f1']:.2f}%")

    # Optionally, log some examples
    for i in range(min(5, len(predictions))):
        logger.info(f"Example {i + 1}:")
        logger.info(f"  - Ground Truth: {ground_truths[i]}")
        logger.info(f"  - Prediction:   {predictions[i]}")


if __name__ == "__main__":
    main()
