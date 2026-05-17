"""
Compute negative log-likelihood (NLL) for true/false datasets and use it for hallucination detection.

Expected CSV format:
- statement: text prompt
- label: 1 for true, 0 for false (optional but required for metrics/threshold)

Outputs a CSV with NLL and optional hallucination predictions.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("Config file not found: %s", path)
        return {}
    except json.JSONDecodeError:
        logging.warning("Config file is not valid JSON: %s", path)
        return {}


def resolve_model_id(model_name: str) -> str:
    opt_sizes = {"350m", "1.3b", "2.7b", "6.7b"}
    if model_name in opt_sizes:
        return f"facebook/opt-{model_name}"
    return model_name


def normalize_dataset_name(dataset_name: str, true_false: bool) -> str:
    if true_false and not dataset_name.endswith("_true_false"):
        return f"{dataset_name}_true_false"
    return dataset_name


def load_dataset(dataset_path: Path, dataset_name: str, true_false: bool) -> pd.DataFrame:
    normalized_name = normalize_dataset_name(dataset_name, true_false)
    dataset_file = dataset_path / f"{normalized_name}.csv"
    return pd.read_csv(dataset_file)


def get_text_column(df: pd.DataFrame) -> str:
    if "statement" in df.columns:
        return "statement"
    if "statements" in df.columns:
        return "statements"
    raise ValueError("CSV must contain a 'statement' column.")


def compute_nll_batch(texts, tokenizer, model, device, max_length=None):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    neg_log_likelihood = -gathered * shift_mask

    token_counts = shift_mask.sum(dim=1).clamp(min=1)
    nll = neg_log_likelihood.sum(dim=1) / token_counts

    return nll.detach().cpu().numpy()


def find_best_threshold(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_threshold = thresholds[0]
    best_acc = -1.0
    for thr in thresholds:
        preds = scores >= thr
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thr
    return best_threshold, best_acc


def parse_torch_dtype(dtype_name: str):
    if dtype_name is None:
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError("dtype must be one of: float16, bfloat16, float32")
    return mapping[dtype_name]


def evaluate_dataset(df: pd.DataFrame, threshold: Optional[float]):
    metrics = {
        "best_threshold": None,
        "accuracy": None,
        "auc": None,
    }
    if "label" not in df.columns:
        return metrics, threshold

    labels = (1 - df["label"].astype(int)).to_numpy()
    scores = df["nll"].to_numpy()
    finite_mask = np.isfinite(scores)
    if not finite_mask.all():
        dropped = int((~finite_mask).sum())
        logging.warning("Dropping %d rows with non-finite NLL for metrics.", dropped)
    labels = labels[finite_mask]
    scores = scores[finite_mask]

    if len(scores) == 0:
        logging.warning("No finite NLL values; skipping metrics.")
        return metrics, threshold

    if threshold is None:
        if len(np.unique(labels)) < 2:
            logging.warning("Only one class present; provide --threshold to label hallucinations.")
        else:
            threshold, best_acc = find_best_threshold(scores, labels)
            metrics["best_threshold"] = float(threshold)
            metrics["accuracy"] = float(best_acc)
            try:
                metrics["auc"] = float(roc_auc_score(labels, scores))
            except ValueError:
                metrics["auc"] = None
    else:
        preds = scores >= threshold
        metrics["best_threshold"] = float(threshold)
        metrics["accuracy"] = float(accuracy_score(labels, preds))
        try:
            metrics["auc"] = float(roc_auc_score(labels, scores))
        except ValueError:
            metrics["auc"] = None

    if threshold is not None:
        preds_full = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
        if "nll" in df.columns:
            finite_mask_full = np.isfinite(df["nll"].to_numpy())
            preds_full.loc[finite_mask_full] = (df.loc[finite_mask_full, "nll"].to_numpy() >= threshold).astype(int)
        df["hallucination_pred"] = preds_full

    return metrics, threshold


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_config("config.json")

    parser = argparse.ArgumentParser(description="Compute NLL and detect hallucinations using true/false datasets.")
    parser.add_argument("--model", help="Model name or OPT size (e.g., 350m, 1.3b, 2.7b, 6.7b).")
    parser.add_argument("--dataset_name", help="Dataset name without .csv extension.")
    parser.add_argument("--dataset_names", nargs='*', help="List of dataset names without .csv extension.")
    parser.add_argument("--dataset_path", help="Path to dataset directory.")
    parser.add_argument("--output_path", help="Path to write output CSV.")
    parser.add_argument("--summary_path", help="Path to write summary metrics CSV.")
    parser.add_argument("--batch_size", type=int, help="Batch size for NLL computation.")
    parser.add_argument("--max_length", type=int, help="Max token length for truncation.")
    parser.add_argument("--true_false", action="store_true", default=None, help="Append '_true_false' to dataset name.")
    parser.add_argument("--remove_period", action="store_true", default=None, help="Strip trailing periods before scoring.")
    parser.add_argument("--threshold", type=float, help="NLL threshold for hallucination prediction.")
    parser.add_argument("--device_map", help="Device map for model loading, e.g. 'auto'.")
    parser.add_argument("--dtype", help="Model dtype: float16, bfloat16, float32.")
    args = parser.parse_args()

    model_name = args.model or config.get("ppl_model") or config.get("model")
    dataset_name = args.dataset_name or config.get("ppl_dataset_name")
    dataset_names = args.dataset_names or config.get("ppl_dataset_names")
    dataset_path = Path(args.dataset_path or config.get("dataset_path", "datasets"))
    output_path = Path(args.output_path or config.get("ppl_output_path", "processed_datasets"))
    summary_path = args.summary_path or config.get("ppl_summary_path")
    batch_size = args.batch_size or config.get("ppl_batch_size") or config.get("batch_size", 8)
    max_length = args.max_length or config.get("ppl_max_length")
    true_false = args.true_false if args.true_false is not None else config.get("true_false", False)
    remove_period = args.remove_period if args.remove_period is not None else config.get("remove_period", False)
    torch_dtype = parse_torch_dtype(args.dtype)
    device_map = args.device_map

    if dataset_names is None:
        if dataset_name is None:
            raise ValueError("Provide --dataset_name/--dataset_names or set 'ppl_dataset_name(s)' in config.json.")
        dataset_names = [dataset_name]
    if model_name is None:
        raise ValueError("Provide --model or set 'ppl_model' or 'model' in config.json.")

    model_id = resolve_model_id(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device_map:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    model.eval()

    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for dataset_name in dataset_names:
        df = load_dataset(dataset_path, dataset_name, true_false)
        text_col = get_text_column(df)
        texts = df[text_col].astype(str).tolist()
        if remove_period:
            texts = [t.rstrip(". ") for t in texts]

        all_nll = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            nll = compute_nll_batch(batch_texts, tokenizer, model, device, max_length)
            all_nll.extend(nll.tolist())

        df["nll"] = np.array(all_nll)

        metrics, _ = evaluate_dataset(df, args.threshold)

        normalized_name = normalize_dataset_name(dataset_name, true_false)
        output_file = output_path / f"{normalized_name}_ppl_predictions.csv"
        df.to_csv(output_file, index=False)

        summary_rows.append({
            "dataset": normalized_name,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "best_threshold": metrics["best_threshold"],
        })

    summary_df = pd.DataFrame(summary_rows)
    if summary_path is None:
        summary_path = str(output_path / "ppl_metrics.csv")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
