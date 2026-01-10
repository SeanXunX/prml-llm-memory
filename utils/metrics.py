import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Computes the F1 score, a measure of word overlap between prediction and ground_truth.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Computes whether the normalized prediction is identical to the normalized ground_truth.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calculate_metrics(predictions: list[str], ground_truths: list[str]) -> dict:
    """
    Calculates average F1 and Exact Match scores for a list of predictions and ground truths.
    """
    f1 = exact_match = total = 0
    for ground_truth, prediction in zip(ground_truths, predictions):
        total += 1
        exact_match += exact_match_score(prediction, ground_truth)
        f1 += f1_score(prediction, ground_truth)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}
