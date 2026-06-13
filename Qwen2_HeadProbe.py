"""
Per-Head Probe Training and Voting Hallucination Detection

Loads selected top-k attention heads from Qwen2_HeadsAttribution.py,
trains baseline probes (SAPLMA, summed-heads) on set_a,
evaluates voting ensemble on set_c.

The per-head classifiers for the selected heads were already trained in
Qwen2_HeadsAttribution.py on set_a. Here we re-train them for consistency
with the same training pipeline (StandardScaler + random restarts), and
add SAPLMA / summed-heads baselines.

Baselines:
  - PPL: negative log-likelihood as hallucination score
  - SAPLMA: hidden state probe (full hidden state at layer 18)
  - Single-head probes: individual head performance
  - Summed-heads: sum of all selected heads' influence vectors
  - Voting ensemble: majority vote across top-k heads

Qwen2-1.5B: 28 layers, 12 heads/layer, head_dim=128, hidden_size=1536
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(config_path="config.json"):
    with open(config_path) as f:
        return json.load(f)


def init_model(model_path: str, dtype=torch.bfloat16, device_map="auto"):
    """Load Qwen2-1.5B model and tokenizer."""
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Override generation config to suppress sampling-related warnings
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False

    print(f"Model loaded. Hidden size={model.config.hidden_size}, "
          f"num_heads={model.config.num_attention_heads}, "
          f"num_layers={model.config.num_hidden_layers}")
    return model, tokenizer


def extract_head_features(model, tokenizer, samples_df, selected_heads,
                          hidden_layer=18, batch_size=4):
    """
    Extract influence vector features at answer last token position
    for the selected attention heads, AND the full hidden state at a
    specified layer for SAPLMA baseline comparison.

    Returns:
        features: dict {head_key: np.array [N, d_model]}
        hidden_features: np.array [N, d_model] — full hidden state at hidden_layer
        labels: np.array [N]
        ppl_scores: np.array [N] — per-sample negative log-likelihood
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    d_model = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device

    # Register hooks on o_proj + hidden state capture
    o_proj_inputs = {}
    hidden_states = {}
    hook_handles = []

    for layer_idx in range(num_layers):
        o_proj_module = model.model.layers[layer_idx].self_attn.o_proj

        def make_hook(l_idx):
            def hook_fn(module, input, output):
                o_proj_inputs[l_idx] = input[0].detach()
            return hook_fn

        handle = o_proj_module.register_forward_hook(make_hook(layer_idx))
        hook_handles.append(handle)

    # Hook for full hidden state (post-layer output) at the target layer.
    # hidden_layer uses 1-based indexing (matching SAPLMA convention where
    # layer=18 means hidden_states[18] = output of model.model.layers[17]).
    target_layer = model.model.layers[hidden_layer - 1]

    def hidden_hook(module, input, output):
        # output[0]: hidden_states after this layer, shape [1, T, d_model]
        hidden_states['layer'] = output[0].detach()

    hidden_handle = target_layer.register_forward_hook(hidden_hook)
    hook_handles.append(hidden_handle)

    # Initialize feature containers
    head_keys = [f"L{layer}_H{head}" for layer, head in selected_heads]
    all_features = {key: [] for key in head_keys}
    all_hidden = []
    all_labels = []
    all_ppl = []

    # Pre-compute W_O for layers referenced by selected_heads (avoids
    # redundant .detach()/.float() casts across samples).
    W_O_cache = {}
    for layer in set(l for l, _ in selected_heads):
        W_O = model.model.layers[layer].self_attn.o_proj.weight.detach()
        W_O_cache[layer] = W_O.float()

    for idx in tqdm(range(len(samples_df)), desc="Extracting features"):
        row = samples_df.iloc[idx]
        question = row['question']
        generated_answer = row['generated_answer']
        label = row['label']

        if not generated_answer.strip():
            continue

        # Build prompt token IDs
        messages = [
            {"role": "system",
             "content": "Follow the given examples and answer the question."},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")
        prompt_length = prompt_ids["input_ids"].shape[1]

        # Use saved generated token IDs when available (avoids decode→encode
        # roundtrip inconsistency). Fall back to text-based tokenization for
        # legacy data that lacks the 'generated_ids' column.
        if 'generated_ids' in row.index and isinstance(row['generated_ids'], str):
            generated_ids_list = json.loads(row['generated_ids'])
            generated_ids = torch.tensor([generated_ids_list], dtype=torch.long, device=device)
            input_ids = torch.cat([prompt_ids["input_ids"].to(device), generated_ids], dim=1)
            total_length = input_ids.shape[1]
        else:
            full_text = prompt_text + generated_answer
            full_inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = full_inputs["input_ids"].to(device)
            total_length = input_ids.shape[1]

        answer_last_pos = total_length - 1
        if answer_last_pos < prompt_length:
            continue

        answer_last_token_id = input_ids[0, answer_last_pos].item()

        # Forward pass with hooks
        o_proj_inputs.clear()
        hidden_states.clear()
        with torch.no_grad():
            outputs = model(input_ids)

        # Compute PPL score (negative log-likelihood of answer tokens)
        logits = outputs.logits  # [1, T, vocab]
        answer_logits = logits[0, prompt_length - 1:total_length - 1, :]
        answer_target_ids = input_ids[0, prompt_length:total_length]
        log_probs = torch.log_softmax(answer_logits.float(), dim=-1)
        nll = -log_probs[range(len(answer_target_ids)), answer_target_ids].mean()
        all_ppl.append(nll.cpu().item())

        # Extract full hidden state at answer position (SAPLMA baseline feature)
        if 'layer' in hidden_states:
            h_state = hidden_states['layer'][0, answer_last_pos].float().cpu().numpy()
            all_hidden.append(h_state)

        # Compute influence vectors for selected heads
        for layer, head in selected_heads:
            key = f"L{layer}_H{head}"
            if layer not in o_proj_inputs:
                all_features[key].append(np.zeros(d_model))
                continue

            attn_out = o_proj_inputs[layer]  # [1, T, num_heads * head_dim]
            W_O = W_O_cache[layer]

            head_out = attn_out[0, answer_last_pos,
                                head * head_dim:(head + 1) * head_dim]
            W_O_h = W_O[:, head * head_dim:(head + 1) * head_dim]
            delta_h = torch.matmul(head_out.float(),
                                   W_O_h.T).cpu().numpy()  # [d_model]
            all_features[key].append(delta_h)

        all_labels.append(label)

    # Remove hooks
    for h in hook_handles:
        h.remove()

    features = {key: np.stack(vals, axis=0) for key, vals in all_features.items()}
    hidden_features = np.stack(all_hidden, axis=0) if all_hidden else None
    labels = np.array(all_labels)
    ppl_scores = np.array(all_ppl)

    return features, hidden_features, labels, ppl_scores


def define_probe(input_dim: int):
    """
    Define the probe: 3-layer feedforward network.
    Architecture: Dense(256, relu) → Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)
    """
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_probe(train_features, train_labels, repeat_each=3, epochs=5):
    """
    Train a binary classifier probe with multiple random restarts.
    Features are standardized (zero mean, unit variance) before training.

    Returns:
        best_model: the probe with highest training accuracy
        scaler: fitted StandardScaler (use to transform test features)
        results: list of (accuracy, auc, threshold) per restart
    """
    # Normalize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    results = []
    best_accuracy = 0
    best_model = None

    for _ in range(repeat_each):
        probe = define_probe(train_features_scaled.shape[1])
        probe.fit(train_features_scaled, train_labels, epochs=epochs,
                  batch_size=32, verbose=0)

        train_pred = probe.predict(train_features_scaled, verbose=0).flatten()
        _, _, thresholds = roc_curve(train_labels, train_pred)
        optimal_threshold = thresholds[
            np.argmax([accuracy_score(train_labels, train_pred > thr)
                       for thr in thresholds])
        ]

        train_acc = accuracy_score(train_labels, train_pred > optimal_threshold)
        fpr, tpr, _ = roc_curve(train_labels, train_pred)
        train_auc = auc(fpr, tpr)

        if train_acc > best_accuracy:
            best_accuracy = train_acc
            best_model = probe

        results.append((train_acc, train_auc, optimal_threshold))

    return best_model, scaler, results


def evaluate_probe(probe, test_features, test_labels, threshold=0.5):
    """Evaluate a probe on test data, return metrics dict."""
    pred_prob = probe.predict(test_features, verbose=0).flatten()
    pred_labels = (pred_prob > threshold).astype(int)

    acc = accuracy_score(test_labels, pred_labels)
    prec = precision_score(test_labels, pred_labels, zero_division=0)
    rec = recall_score(test_labels, pred_labels, zero_division=0)
    f1 = f1_score(test_labels, pred_labels, zero_division=0)

    fpr, tpr, _ = roc_curve(test_labels, pred_prob)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': roc_auc,
        'threshold': threshold,
    }


def voting_prediction(probes, features_dict, selected_heads, threshold=0.5):
    """
    Majority voting across k single-head classifiers.

    Each probe outputs a probability; convert to binary (> threshold),
    then take majority vote (>= 3 out of 5).

    Returns:
        predictions: np.array [N] — final prediction (1=correct, 0=hallucination)
        individual_preds: dict {head_key: np.array [N]}
    """
    individual_preds = {}
    for layer, head in selected_heads:
        key = f"L{layer}_H{head}"
        if key in probes and key in features_dict:
            prob = probes[key].predict(features_dict[key], verbose=0).flatten()
            individual_preds[key] = (prob > threshold).astype(int)

    if not individual_preds:
        return np.array([]), {}

    pred_matrix = np.stack(list(individual_preds.values()), axis=0)  # [k, N]
    vote_count = pred_matrix.sum(axis=0)  # [N]
    k = len(selected_heads)
    majority = (k + 1) // 2  # e.g., 3 for k=5
    predictions = (vote_count >= majority).astype(int)

    return predictions, individual_preds


def evaluate_ppl(ppl_scores, labels):
    """
    Evaluate PPL-based hallucination detection.

    ppl_scores: negative log-likelihood per sample (higher = more uncertain).
    We negate to get a "confidence" score: higher neg_ppl → more likely correct.
    Threshold is selected by maximizing accuracy (consistent with probe evaluation).
    """
    # Negate NLL: higher value → lower uncertainty → more likely correct
    neg_ppl = -ppl_scores
    fpr, tpr, thresholds = roc_curve(labels, neg_ppl)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold by maximizing accuracy (same strategy as probes)
    optimal_threshold = thresholds[
        np.argmax([accuracy_score(labels, neg_ppl > thr)
                   for thr in thresholds])
    ]
    pred_labels = (neg_ppl > optimal_threshold).astype(int)

    acc = accuracy_score(labels, pred_labels)
    prec = precision_score(labels, pred_labels, zero_division=0)
    rec = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': roc_auc,
        'threshold': float(optimal_threshold),
    }


def plot_comparison(methods_metrics, save_path):
    """Plot bar chart comparing different methods."""
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    method_names = list(methods_metrics.keys())

    x = np.arange(len(metrics_names))
    width = 0.8 / len(method_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(method_names):
        values = [methods_metrics[method].get(m, 0) for m in metrics_names]
        bars = ax.bar(x + i * width, values, width, label=method)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Hallucination Detection Performance Comparison')
    ax.set_xticks(x + width * (len(method_names) - 1) / 2)
    ax.set_xticklabels(metrics_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def run_probe_experiment(config_path="config.json",
                         output_dir=None,
                         top_k=5,
                         repeat_each=3,
                         epochs=5,
                         batch_size=4,
                         hidden_layer=18):
    """Run the full head probe training and voting evaluation pipeline."""
    config = load_config(config_path)

    model_path = config.get("model_path", "model")
    dtype_str = config.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device_map = config.get("device_map", "auto")
    processed_path = Path(config.get("processed_dataset_path", "processed_datasets"))

    if output_dir is None:
        output_dir = processed_path
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load selected heads (from classifier-based selection in HeadsAttribution)
    top_heads_file = processed_path / "top5_heads.json"
    with open(top_heads_file) as f:
        top_heads_info = json.load(f)
    selected_heads = [tuple(h) for h in top_heads_info["selected_heads"]]
    print(f"Loaded {len(selected_heads)} selected heads: {selected_heads}")
    if "head_details" in top_heads_info:
        for key, detail in top_heads_info["head_details"].items():
            print(f"  {key}: AUC={detail.get('auc', 'N/A')}")

    # Load model
    model, tokenizer = init_model(model_path, dtype=dtype, device_map=device_map)

    # Load set_a (training) and set_c (evaluation)
    set_a_path = processed_path / "triviaqa_set_a.csv"
    set_c_path = processed_path / "triviaqa_set_c.csv"

    set_a_df = pd.read_csv(set_a_path)
    set_c_df = pd.read_csv(set_c_path)
    print(f"Set A: {len(set_a_df)} samples (training)")
    print(f"Set C: {len(set_c_df)} samples (evaluation)")

    # Step 1: Extract features for set_a
    print("\n=== Step 1: Extracting features for set A (training) ===")
    train_features, train_hidden, train_labels, train_ppl = extract_head_features(
        model, tokenizer, set_a_df, selected_heads,
        hidden_layer=hidden_layer, batch_size=batch_size
    )

    # Step 2: Train per-head probes + SAPLMA baseline
    print("\n=== Step 2: Training per-head probes ===")
    probes = {}
    scalers = {}
    probe_train_metrics = {}

    for layer, head in selected_heads:
        key = f"L{layer}_H{head}"
        print(f"\n  Training probe for {key}...")
        feat = train_features[key]

        best_probe, scaler, results = train_probe(feat, train_labels,
                                                   repeat_each=repeat_each,
                                                   epochs=epochs)
        probes[key] = best_probe
        scalers[key] = scaler

        avg_acc = np.mean([r[0] for r in results])
        avg_auc = np.mean([r[1] for r in results])
        avg_thr = np.mean([r[2] for r in results])
        probe_train_metrics[key] = {
            'avg_accuracy': avg_acc,
            'avg_auc': avg_auc,
            'avg_threshold': avg_thr,
        }
        print(f"    Train: Acc={avg_acc:.4f}, AUC={avg_auc:.4f}, Thr={avg_thr:.4f}")

    # Train SAPLMA baseline probe (full hidden state)
    print("\n  Training SAPLMA baseline probe (full hidden state)...")
    saplma_probe, saplma_scaler, saplma_results = train_probe(
        train_hidden, train_labels, repeat_each=repeat_each, epochs=epochs
    )
    saplma_avg_acc = np.mean([r[0] for r in saplma_results])
    saplma_avg_auc = np.mean([r[1] for r in saplma_results])
    saplma_avg_thr = np.mean([r[2] for r in saplma_results])
    print(f"    SAPLMA Train: Acc={saplma_avg_acc:.4f}, AUC={saplma_avg_auc:.4f}, "
          f"Thr={saplma_avg_thr:.4f}")

    # Train summed-heads probe (sum of all selected heads' influence vectors)
    print("\n  Training summed-heads probe...")
    train_summed = sum(train_features[f"L{l}_H{h}"] for l, h in selected_heads)
    summed_probe, summed_scaler, summed_results = train_probe(
        train_summed, train_labels, repeat_each=repeat_each, epochs=epochs
    )
    summed_avg_acc = np.mean([r[0] for r in summed_results])
    summed_avg_auc = np.mean([r[1] for r in summed_results])
    summed_avg_thr = np.mean([r[2] for r in summed_results])
    print(f"    Summed Train: Acc={summed_avg_acc:.4f}, AUC={summed_avg_auc:.4f}, "
          f"Thr={summed_avg_thr:.4f}")

    # Step 3: Extract features for set_c (evaluation)
    print("\n=== Step 3: Extracting features for set C (evaluation) ===")
    test_features, test_hidden, test_labels, test_ppl = extract_head_features(
        model, tokenizer, set_c_df, selected_heads,
        hidden_layer=hidden_layer, batch_size=batch_size
    )

    # Free model
    del model, tokenizer
    torch.cuda.empty_cache()

    # Step 4: Evaluate individual probes and voting
    print("\n=== Step 4: Evaluation on set C ===")
    all_metrics = {}

    # PPL baseline
    ppl_metrics = evaluate_ppl(test_ppl, test_labels)
    all_metrics['PPL'] = ppl_metrics
    print(f"\nPPL Baseline:")
    print(f"  Acc={ppl_metrics['accuracy']:.4f}, AUC={ppl_metrics['auc']:.4f}, "
          f"F1={ppl_metrics['f1']:.4f}")

    # SAPLMA baseline (full hidden state)
    test_hidden_scaled = saplma_scaler.transform(test_hidden)
    saplma_metrics = evaluate_probe(saplma_probe, test_hidden_scaled,
                                     test_labels, threshold=saplma_avg_thr)
    all_metrics['SAPLMA-L18'] = saplma_metrics
    print(f"\nSAPLMA Baseline (L18 hidden state):")
    print(f"  Acc={saplma_metrics['accuracy']:.4f}, AUC={saplma_metrics['auc']:.4f}, "
          f"F1={saplma_metrics['f1']:.4f}")

    # Summed-heads probe (sum of all selected heads' influence vectors)
    test_summed = sum(test_features[f"L{l}_H{h}"] for l, h in selected_heads)
    test_summed_scaled = summed_scaler.transform(test_summed)
    summed_metrics = evaluate_probe(summed_probe, test_summed_scaled,
                                     test_labels, threshold=summed_avg_thr)
    all_metrics['Summed-Heads'] = summed_metrics
    print(f"\nSummed Heads ({len(selected_heads)} heads summed):")
    print(f"  Acc={summed_metrics['accuracy']:.4f}, AUC={summed_metrics['auc']:.4f}, "
          f"F1={summed_metrics['f1']:.4f}")

    # Individual probes (apply training scaler to test features)
    for layer, head in selected_heads:
        key = f"L{layer}_H{head}"
        threshold = probe_train_metrics[key]['avg_threshold']
        test_feat_scaled = scalers[key].transform(test_features[key])
        metrics = evaluate_probe(probes[key], test_feat_scaled,
                                 test_labels, threshold=threshold)
        all_metrics[f'Single-{key}'] = metrics
        print(f"Single {key}: Acc={metrics['accuracy']:.4f}, "
              f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

    # Voting ensemble
    print("\n  Running voting ensemble...")
    # Use average threshold from individual probes
    avg_threshold = np.mean([probe_train_metrics[f"L{l}_H{h}"]['avg_threshold']
                             for l, h in selected_heads])

    # Apply scalers to test features for voting
    test_features_scaled = {
        f"L{l}_H{h}": scalers[f"L{l}_H{h}"].transform(test_features[f"L{l}_H{h}"])
        for l, h in selected_heads
        if f"L{l}_H{h}" in test_features
    }
    vote_preds, individual_preds = voting_prediction(
        probes, test_features_scaled, selected_heads, threshold=avg_threshold
    )

    if len(vote_preds) > 0:
        vote_acc = accuracy_score(test_labels, vote_preds)
        vote_prec = precision_score(test_labels, vote_preds, zero_division=0)
        vote_rec = recall_score(test_labels, vote_preds, zero_division=0)
        vote_f1 = f1_score(test_labels, vote_preds, zero_division=0)

        # For voting AUC, use average probability across probes
        avg_prob = np.zeros(len(test_labels))
        count = 0
        for layer, head in selected_heads:
            key = f"L{layer}_H{head}"
            if key in probes and key in test_features_scaled:
                prob = probes[key].predict(test_features_scaled[key], verbose=0).flatten()
                avg_prob += prob
                count += 1
        avg_prob /= max(count, 1)
        fpr, tpr, _ = roc_curve(test_labels, avg_prob)
        vote_auc = auc(fpr, tpr)

        all_metrics['Voting-Ensemble'] = {
            'accuracy': vote_acc,
            'precision': vote_prec,
            'recall': vote_rec,
            'f1': vote_f1,
            'auc': vote_auc,
        }
        print(f"Voting Ensemble: Acc={vote_acc:.4f}, AUC={vote_auc:.4f}, "
              f"F1={vote_f1:.4f}")

    # Step 5: Save results
    print("\n=== Step 5: Saving results ===")

    # Save metrics
    metrics_rows = []
    for method, m in all_metrics.items():
        row = {'method': method}
        row.update(m)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_file = output_dir / "head_probe_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")

    # Save probes
    probes_dir = output_dir / "head_probes"
    probes_dir.mkdir(exist_ok=True)
    for key, probe in probes.items():
        probe.save(probes_dir / f"{key}_probe.h5")
    saplma_probe.save(probes_dir / "SAPLMA_L18_probe.h5")
    summed_probe.save(probes_dir / "Summed_Heads_probe.h5")
    print(f"Probes saved to {probes_dir}")

    # Plot comparison
    fig_path = output_dir / "fig_voting_comparison.png"
    plot_comparison(all_metrics, fig_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("=== Summary ===")
    print(f"{'Method':<30} {'Acc':>8} {'Prec':>8} {'Recall':>8} "
          f"{'F1':>8} {'AUC':>8}")
    print("-" * 70)
    for method, m in all_metrics.items():
        print(f"{method:<30} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    return all_metrics, probes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-head probe training and voting hallucination detection"
    )
    parser.add_argument("--config", default="config.json",
                        help="Path to config file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top heads (should match HeadsAttribution)")
    parser.add_argument("--repeat_each", type=int, default=3,
                        help="Random restarts per probe")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per probe")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for feature extraction")
    parser.add_argument("--hidden_layer", type=int, default=18,
                        help="Layer for SAPLMA baseline hidden state (default 18)")
    args = parser.parse_args()

    run_probe_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        top_k=args.top_k,
        repeat_each=args.repeat_each,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_layer=args.hidden_layer,
    )
