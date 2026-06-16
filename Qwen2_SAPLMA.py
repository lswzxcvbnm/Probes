"""
SAPLMA Experiment with Qwen2-1.5B

Reproduces the SAPLMA paper (Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying")
using the Qwen2-1.5B model. Extracts embeddings from intermediate layers to the last layer
(every 4 layers) and trains probes for each layer.

Qwen2-1.5B has 28 transformer layers (hidden_states[0]=embedding, [1]-[28]=layers 1-28).
We test layers: 14(middle), 18, 22, 26, 28(last) — every 4 layers from middle to last.

Paper methodology:
- Feedforward probe: 256→128→64→1 with ReLU, sigmoid output, Adam, binary cross-entropy
- Leave-one-out: train on 5 topics, test on 1
- 3 random restarts per run, report mean accuracy/AUC
- Extract last token's hidden state as embedding
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import os
from copy import deepcopy
from tqdm import tqdm
import argparse

from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qwen2_saplma.log'
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.json"):
    with open(config_path) as f:
        return json.load(f)


def init_model(model_path: str, dtype=torch.bfloat16, device_map="auto"):
    """Load Qwen2-1.5B model and tokenizer from local path."""
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded. Hidden size={model.config.hidden_size}, "
          f"num_layers={model.config.num_hidden_layers}")
    return model, tokenizer


def load_dataset(dataset_path: Path, dataset_name: str):
    """Load a true_false dataset CSV."""
    file_path = dataset_path / f"{dataset_name}_true_false.csv"
    df = pd.read_csv(file_path)
    return df


def extract_embeddings(model, tokenizer, statements: list, layer: int,
                       batch_size: int = 8, remove_period: bool = True):
    """
    Extract embeddings from a specific layer for a list of statements.
    Uses the last token's hidden state as the embedding (SAPLMA methodology).

    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        statements: List of statement strings.
        layer: Layer index (1-based, e.g., 28 = last layer).
        batch_size: Batch size for processing.
        remove_period: Whether to strip trailing periods.

    Returns:
        numpy array of shape (num_statements, hidden_size).
    """
    all_embeddings = []
    device = next(model.parameters()).device

    if remove_period:
        statements = [s.rstrip(". ") for s in statements]

    num_batches = (len(statements) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Extracting layer {layer}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(statements))
        batch = statements[start:end]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # hidden_states[layer] is the output of transformer layer `layer`
        # hidden_states[0] = embedding output, [1]..[28] = transformer layers
        hidden_states = outputs.hidden_states[layer]

        # Find last real token position for each sequence
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1

        for i in range(len(batch)):
            emb = hidden_states[i, seq_lengths[i], :].detach().cpu().float().numpy()
            all_embeddings.append(emb)

    return np.array(all_embeddings)


def define_probe(input_dim: int):
    """
    Define the SAPLMA probe: 3-layer feedforward network.
    Architecture: 256→128→64→1, ReLU activations, sigmoid output.
    """
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
    """
    Train probe and evaluate on test set. Repeat with random restarts.

    Returns:
        results: list of (accuracy, auc, optimal_threshold)
        best_model: the best probe model
    """
    results = []
    best_accuracy = 0
    best_model = None

    for i in range(repeat_each):
        probe = define_probe(train_emb.shape[1])

        # Train (paper uses 5 epochs)
        probe.fit(train_emb, train_labels, epochs=epochs, batch_size=32, verbose=0)

        # Find optimal threshold on training set
        train_pred = probe.predict(train_emb, verbose=0)
        fpr_val, tpr_val, thresholds_val = roc_curve(train_labels, train_pred)
        optimal_threshold = thresholds_val[
            np.argmax([accuracy_score(train_labels, train_pred > thr)
                       for thr in thresholds_val])
        ]

        # Evaluate on test set
        test_pred = probe.predict(test_emb, verbose=0)
        test_accuracy = accuracy_score(test_labels, test_pred > optimal_threshold)

        # ROC AUC
        fpr, tpr, _ = roc_curve(test_labels, test_pred)
        roc_auc = auc(fpr, tpr)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = probe

        results.append((test_accuracy, roc_auc, optimal_threshold))

    return results, best_model


def run_saplma_experiment(config_path="config.json"):
    """Run the full SAPLMA experiment on Qwen2-1.5B."""
    config = load_config(config_path)

    # Model configuration
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

    # Qwen2-1.5B has 28 layers. Test every 2 layers from middle to last:
    # middle=14, then 16, 18, 20, 22, 24, 26, 28 (last)
    num_layers = 28
    middle = num_layers // 2
    layers_to_test = list(range(middle, num_layers, 2)) + [num_layers]
    # Result: [14, 16, 18, 20, 22, 24, 26, 28]

    print(f"SAPLMA Experiment with {model_alias}")
    print(f"Layers to test: {layers_to_test}")
    print(f"Datasets: {dataset_names}")
    print(f"Leave-one-out cross-validation, {repeat_each} restarts, {epochs} epochs")
    print("=" * 70)

    # Load model
    model, tokenizer = init_model(model_path)

    # Step 1: Extract embeddings for all datasets and layers
    print("\n=== Step 1: Extracting Embeddings ===")
    embeddings_cache = {}  # {(dataset_name, layer): numpy_array}

    for dataset_name in tqdm(dataset_names, desc="Datasets"):
        df = load_dataset(dataset_path, dataset_name)
        statements = df['statement'].tolist()
        labels = df['label'].values

        for layer in tqdm(layers_to_test, desc=f"Layers for {dataset_name}", leave=False):
            emb = extract_embeddings(model, tokenizer, statements, layer,
                                     batch_size=batch_size, remove_period=remove_period)
            embeddings_cache[(dataset_name, layer)] = emb

            # Save embeddings to CSV (for reproducibility)
            save_df = df.copy()
            save_df['embeddings'] = [emb[i].tolist() for i in range(len(emb))]
            output_path.mkdir(parents=True, exist_ok=True)
            save_file = output_path / f"embeddings_{dataset_name}{model_alias}_{layer}_rmv_period.csv"
            save_df.to_csv(save_file, index=False)

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()

    # Step 2: Train probes with leave-one-out cross-validation
    print("\n=== Step 2: Training Probes (Leave-One-Out) ===")
    all_metrics = []

    for layer in layers_to_test:
        print(f"\n--- Layer {layer} ---")

        for test_idx, test_name in enumerate(dataset_names):
            # Prepare test data
            test_emb = embeddings_cache[(test_name, layer)]
            test_df = load_dataset(dataset_path, test_name)
            test_labels = test_df['label'].values

            # Prepare training data (all other topics)
            train_names = [n for n in dataset_names if n != test_name]
            train_emb_list = [embeddings_cache[(n, layer)] for n in train_names]
            train_emb = np.concatenate(train_emb_list, axis=0)

            train_label_list = []
            for n in train_names:
                df = load_dataset(dataset_path, n)
                train_label_list.append(df['label'].values)
            train_labels = np.concatenate(train_label_list, axis=0)

            # Train and evaluate
            results, best_model = train_and_evaluate(
                train_emb, train_labels, test_emb, test_labels,
                repeat_each=repeat_each, epochs=epochs
            )

            # Aggregate results
            acc_list = [r[0] for r in results]
            auc_list = [r[1] for r in results]
            thr_list = [r[2] for r in results]

            avg_acc = np.mean(acc_list)
            avg_auc = np.mean(auc_list)
            avg_thr = np.mean(thr_list)

            print(f"  Test on {test_name:15s} | "
                  f"Acc: {avg_acc:.4f} | AUC: {avg_auc:.4f} | Threshold: {avg_thr:.4f}")

            all_metrics.append({
                "layer": layer,
                "test_topic": test_name,
                "train_topics": ",".join(train_names),
                "avg_accuracy": round(avg_acc, 4),
                "avg_auc": round(avg_auc, 4),
                "avg_threshold": round(avg_thr, 4),
            })

            # Save best probe
            if config.get("save_probes", True):
                probes_path.mkdir(parents=True, exist_ok=True)
                probe_file = probes_path / f"{model_alias}_{layer}_{test_name}_rp.h5"
                best_model.save(probe_file)

    # Step 3: Print summary
    print("\n" + "=" * 70)
    print("=== Results Summary ===")
    metrics_df = pd.DataFrame(all_metrics)

    # Per-layer average
    print("\nAverage accuracy per layer (across all test topics):")
    for layer in layers_to_test:
        layer_df = metrics_df[metrics_df['layer'] == layer]
        avg_acc = layer_df['avg_accuracy'].mean()
        avg_auc = layer_df['avg_auc'].mean()
        print(f"  Layer {layer:2d} | Avg Acc: {avg_acc:.4f} | Avg AUC: {avg_auc:.4f}")

    # Per-topic average
    print("\nAverage accuracy per topic (across all layers):")
    for topic in dataset_names:
        topic_df = metrics_df[metrics_df['test_topic'] == topic]
        avg_acc = topic_df['avg_accuracy'].mean()
        avg_auc = topic_df['avg_auc'].mean()
        print(f"  {topic:15s} | Avg Acc: {avg_acc:.4f} | Avg AUC: {avg_auc:.4f}")

    # Save metrics
    metrics_file = output_path / "qwen2_saplma_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAPLMA experiment with Qwen2-1.5B")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    run_saplma_experiment(args.config)
