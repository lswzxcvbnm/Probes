"""
Token Position Experiment with Qwen2-1.5B Layer 16

Explores how different token positions affect SAPLMA probe performance.
Uses Qwen2-1.5B layer 16 (best layer from Stage 1), extracting embeddings from:
  - Position 0 (first token)
  - Position 1 (second token)
  - Position -2 (second to last token)
  - Position -1 (last token, i.e. the default SAPLMA setting)

Then trains probes for each position and compares results.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import argparse
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qwen2_token_position.log'
)
logger = logging.getLogger(__name__)

POSITION_NAMES = {
    0: "first",
    1: "second",
    -2: "second_to_last",
    -1: "last",
}


def load_config(config_path="config.json"):
    with open(config_path) as f:
        return json.load(f)


def init_model(model_path: str, dtype=torch.bfloat16, device_map="auto"):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded. Hidden size={model.config.hidden_size}, "
          f"num_layers={model.config.num_hidden_layers}")
    return model, tokenizer


def load_dataset(dataset_path: Path, dataset_name: str):
    file_path = dataset_path / f"{dataset_name}_true_false.csv"
    return pd.read_csv(file_path)


def extract_embeddings_by_position(model, tokenizer, statements: list, layer: int,
                                    positions: list, batch_size: int = 8,
                                    remove_period: bool = True):
    """
    Extract embeddings from multiple token positions at a given layer.

    Args:
        positions: list of int positions. Positive = from start, negative = from end.
                   e.g. [0, 1, -2, -1]

    Returns:
        dict mapping position -> numpy array of shape (num_statements, hidden_size)
    """
    if remove_period:
        statements = [s.rstrip(". ") for s in statements]

    device = next(model.parameters()).device
    all_embeddings = {pos: [] for pos in positions}
    num_batches = (len(statements) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Layer {layer}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(statements))
        batch = statements[start:end]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, hidden)
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # last real token index

        for i in range(len(batch)):
            sl = seq_lengths[i].item()
            for pos in positions:
                if pos >= 0:
                    # Positive: absolute index from start
                    idx = pos
                else:
                    # Negative: relative to end (e.g., -1 = last token)
                    idx = sl + pos + 1

                # Clamp to valid range [0, sl]
                idx = max(0, min(idx, sl))
                emb = hidden_states[i, idx, :].detach().cpu().float().numpy()
                all_embeddings[pos].append(emb)

    return {pos: np.array(embs) for pos, embs in all_embeddings.items()}


def define_probe(input_dim: int):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(train_emb, train_labels, test_emb, test_labels,
                       repeat_each=3, epochs=5):
    results = []
    best_accuracy = 0
    best_model = None

    for _ in range(repeat_each):
        probe = define_probe(train_emb.shape[1])
        probe.fit(train_emb, train_labels, epochs=epochs, batch_size=32, verbose=0)

        train_pred = probe.predict(train_emb, verbose=0)
        _, _, thresholds_val = roc_curve(train_labels, train_pred)
        optimal_threshold = thresholds_val[
            np.argmax([accuracy_score(train_labels, train_pred > thr)
                       for thr in thresholds_val])
        ]

        test_pred = probe.predict(test_emb, verbose=0)
        test_accuracy = accuracy_score(test_labels, test_pred > optimal_threshold)
        fpr, tpr, _ = roc_curve(test_labels, test_pred)
        roc_auc = auc(fpr, tpr)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = probe

        results.append((test_accuracy, roc_auc, optimal_threshold))

    return results, best_model


def run_position_experiment(config_path="config.json"):
    config = load_config(config_path)

    model_path = config.get("model_path", "model")
    model_alias = config.get("model_alias", "qwen2_1.5b")
    remove_period = config.get("remove_period", True)
    dataset_names = config["list_of_datasets"]
    dataset_path = Path(config["dataset_path"])
    output_path = Path(config["processed_dataset_path"])
    probes_path = Path(config.get("probes_dir", "probes"))
    repeat_each = config.get("repeat_each", 3)
    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 8)

    LAYER = 16  # Best layer from Stage 1
    positions = [0, 1, -2, -1]

    print(f"Token Position Experiment — {model_alias}, Layer {LAYER}")
    print(f"Positions: {positions} ({', '.join(POSITION_NAMES[p] for p in positions)})")
    print(f"Datasets: {dataset_names}")
    print(f"Leave-one-out, {repeat_each} restarts, {epochs} epochs")
    print("=" * 70)

    model, tokenizer = init_model(model_path)

    # Step 1: Extract embeddings for all datasets at all positions
    print("\n=== Step 1: Extracting Embeddings ===")
    embeddings_cache = {}  # {(dataset_name, position): numpy_array}

    for dataset_name in tqdm(dataset_names, desc="Datasets"):
        df = load_dataset(dataset_path, dataset_name)
        statements = df['statement'].tolist()

        pos_embeddings = extract_embeddings_by_position(
            model, tokenizer, statements, LAYER, positions,
            batch_size=batch_size, remove_period=remove_period
        )

        for pos in positions:
            embeddings_cache[(dataset_name, pos)] = pos_embeddings[pos]

            # Save to CSV
            save_df = df.copy()
            save_df['embeddings'] = [pos_embeddings[pos][i].tolist() for i in range(len(pos_embeddings[pos]))]
            output_path.mkdir(parents=True, exist_ok=True)
            fname = f"embeddings_{dataset_name}{model_alias}_L{LAYER}_{POSITION_NAMES[pos]}_rmv_period.csv"
            save_df.to_csv(output_path / fname, index=False)

    del model, tokenizer
    torch.cuda.empty_cache()

    # Step 2: Train probes — leave-one-out for each position
    print("\n=== Step 2: Training Probes ===")
    all_metrics = []

    for pos in positions:
        pos_name = POSITION_NAMES[pos]
        print(f"\n--- Position: {pos_name} (index {pos}) ---")

        for test_name in dataset_names:
            test_emb = embeddings_cache[(test_name, pos)]
            test_labels = load_dataset(dataset_path, test_name)['label'].values

            train_names = [n for n in dataset_names if n != test_name]
            train_emb = np.concatenate([embeddings_cache[(n, pos)] for n in train_names])
            train_labels = np.concatenate([load_dataset(dataset_path, n)['label'].values for n in train_names])

            results, best_model = train_and_evaluate(
                train_emb, train_labels, test_emb, test_labels,
                repeat_each=repeat_each, epochs=epochs,
            )

            acc_list = [r[0] for r in results]
            auc_list = [r[1] for r in results]
            thr_list = [r[2] for r in results]

            avg_acc = np.mean(acc_list)
            avg_auc = np.mean(auc_list)
            avg_thr = np.mean(thr_list)

            print(f"  Test on {test_name:15s} | "
                  f"Acc: {avg_acc:.4f} | AUC: {avg_auc:.4f} | Threshold: {avg_thr:.4f}")

            all_metrics.append({
                "position": pos_name,
                "position_index": pos,
                "layer": LAYER,
                "test_topic": test_name,
                "avg_accuracy": round(avg_acc, 4),
                "avg_auc": round(avg_auc, 4),
                "avg_threshold": round(avg_thr, 4),
            })

            if config.get("save_probes", True):
                probes_path.mkdir(parents=True, exist_ok=True)
                probe_file = probes_path / f"{model_alias}_L{LAYER}_{pos_name}_{test_name}_rp.h5"
                best_model.save(probe_file)

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("=== Results Summary ===")
    metrics_df = pd.DataFrame(all_metrics)

    print(f"\n{'Position':<20} {'Avg Acc':>10} {'Avg AUC':>10}")
    print("-" * 42)
    for pos in positions:
        pos_name = POSITION_NAMES[pos]
        pos_df = metrics_df[metrics_df['position'] == pos_name]
        avg_acc = pos_df['avg_accuracy'].mean()
        avg_auc = pos_df['avg_auc'].mean()
        print(f"{pos_name:<20} {avg_acc:>10.4f} {avg_auc:>10.4f}")

    print(f"\nPer-topic breakdown:")
    print(f"{'Topic':<15}", end="")
    for pos in positions:
        pos_name = POSITION_NAMES[pos]
        print(f"  {pos_name+'_acc':>16}", end="")
    print()
    print("-" * (15 + 18 * len(positions)))
    for topic in dataset_names:
        print(f"{topic:<15}", end="")
        for pos in positions:
            pos_name = POSITION_NAMES[pos]
            row = metrics_df[(metrics_df['position'] == pos_name) & (metrics_df['test_topic'] == topic)]
            acc = row['avg_accuracy'].values[0]
            print(f"  {acc:>16.4f}", end="")
        print()

    metrics_file = output_path / "qwen2_token_position_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token position experiment with Qwen2-1.5B")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    run_position_experiment(args.config)
