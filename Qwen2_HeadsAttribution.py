"""
Attention Head Logit Attribution for Qwen2-1.5B — Classifier-Based Head Selection

For each of the 336 attention heads (28 layers × 12 heads), trains an independent
binary classifier using the head's influence vector at the answer's last token
position as feature. Selects the top-k heads whose classifiers achieve the highest
AUC on a held-out validation set.

Pipeline:
  1. Load set_a (training) and set_b (validation) from Qwen2_TriviaQA_generate.py
  2. Pass 1: extract influence vector features for ALL 336 heads on both sets
  3. Train 336 binary classifiers on set_a, evaluate AUC on set_b
  4. Select top-k heads by validation AUC
  5. Save selected head classifiers, AUC rankings, and features
  6. Visualize AUC heatmap across all heads

Qwen2-1.5B architecture:
  - 28 layers, 12 attention heads per layer, head_dim=128, hidden_size=1536
  - Total heads: 28 * 12 = 336
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


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


def register_o_proj_hooks_for_contributions(model, W_U, sample_contribs):
    """
    Register hooks on each layer's o_proj that directly compute per-head
    logit contributions at a given position, without storing full activations.

    Args:
        model: the loaded model
        W_U: np.array [d_model, vocab] — unembedding matrix (on CPU)
        sample_contribs: list — will be filled with {head_key: float} per sample

    Returns:
        hook_handles: list of hook handles (for cleanup)
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    hook_handles = []

    for layer_idx in range(model.config.num_hidden_layers):
        o_proj_module = model.model.layers[layer_idx].self_attn.o_proj

        def make_hook(l_idx):
            def hook_fn(module, input, output):
                # input[0]: [1, T, num_heads * head_dim] — concatenated head outputs
                # Compute per-head contributions at the answer position
                attn_input = input[0][0]  # [T, num_heads * head_dim], on GPU
                W_O = module.weight.detach()  # [d_model, num_heads * head_dim], on GPU

                for h in range(num_heads):
                    head_out = attn_input[:, h * head_dim:(h + 1) * head_dim]  # [T, head_dim]
                    W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]  # [d_model, head_dim]
                    # delta_h at all positions: [T, d_model]
                    delta = torch.matmul(head_out.float(), W_O_h.float().T)
                    # Store as CPU numpy — keyed by (layer, head)
                    sample_contribs[(l_idx, h)] = delta.cpu().numpy()
            return hook_fn

        handle = o_proj_module.register_forward_hook(make_hook(layer_idx))
        hook_handles.append(handle)

    return hook_handles


def register_o_proj_hooks_for_selected_heads(model, selected_heads):
    """
    Register hooks on each layer's o_proj to capture its INPUT only for
    layers that contain selected heads.

    Args:
        model: the loaded model
        selected_heads: list of (layer, head)

    Returns:
        hook_handles: list of hook handles
        o_proj_inputs: dict {layer_idx: tensor on CPU}
    """
    # Only hook layers that have selected heads
    layers_needed = set(l for l, h in selected_heads)
    o_proj_inputs = {}
    hook_handles = []

    for layer_idx in layers_needed:
        o_proj_module = model.model.layers[layer_idx].self_attn.o_proj

        def make_hook(l_idx):
            def hook_fn(module, input, output):
                o_proj_inputs[l_idx] = input[0][0].detach().float().cpu().numpy()  # [T, D]
            return hook_fn

        handle = o_proj_module.register_forward_hook(make_hook(layer_idx))
        hook_handles.append(handle)

    print(f"Registered hooks on {len(hook_handles)} o_proj modules "
          f"(selected head layers only)")
    return hook_handles, o_proj_inputs


def compute_contributions_for_sample(o_proj_inputs, W_U, model,
                                     answer_last_pos, answer_token_id):
    """
    Compute per-head logit contributions at answer_last_pos from stored
    o_proj inputs.

    Args:
        o_proj_inputs: dict {layer_idx: np.array [T, num_heads * head_dim]}
        W_U: np.array [d_model, vocab]
        model: the model (for architecture params)
        answer_last_pos: int — position of answer's last token
        answer_token_id: int — token ID of answer's last token

    Returns:
        contribs: dict {(layer, head): float}
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    contribs = {}

    target_vec = W_U[:, answer_token_id]  # [d_model]

    for layer_idx, attn_out in o_proj_inputs.items():
        W_O = model.model.layers[layer_idx].self_attn.o_proj.weight.detach()
        W_O_np = W_O.float().cpu().numpy()  # [d_model, num_heads * head_dim]

        head_out = attn_out[answer_last_pos]  # [num_heads * head_dim]
        for h in range(num_heads):
            head_slice = head_out[h * head_dim:(h + 1) * head_dim]  # [head_dim]
            W_O_h = W_O_np[:, h * head_dim:(h + 1) * head_dim]  # [d_model, head_dim]
            delta_h = head_slice @ W_O_h.T  # [d_model]
            contribs[(layer_idx, h)] = float(delta_h @ target_vec)

    return contribs


def compute_influence_vectors_selected(selected_heads, o_proj_inputs, model):
    """
    Compute influence vectors only for selected heads at all positions.

    Args:
        selected_heads: list of (layer, head)
        o_proj_inputs: dict {layer_idx: np.array [T, num_heads * head_dim]}
        model: the model (for o_proj weights)

    Returns:
        influences: dict {(layer, head): np.array [T, d_model]}
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    influences = {}

    for layer, head in selected_heads:
        if layer not in o_proj_inputs:
            continue
        attn_out = o_proj_inputs[layer]  # [T, num_heads * head_dim]
        W_O = model.model.layers[layer].self_attn.o_proj.weight.detach().float().cpu().numpy()
        W_O_h = W_O[:, head * head_dim:(head + 1) * head_dim]  # [d_model, head_dim]
        head_out = attn_out[:, head * head_dim:(head + 1) * head_dim]  # [T, head_dim]
        delta = head_out @ W_O_h.T  # [T, d_model]
        influences[(layer, head)] = delta

    return influences


def compute_logit_contributions(contributions_per_sample, answer_token_ids,
                                 answer_last_positions):
    """
    Aggregate per-sample scalar contributions into arrays per head.

    Args:
        contributions_per_sample: list[dict {(layer, head): float}]
        answer_token_ids: list[int]
        answer_last_positions: list[int]

    Returns:
        contributions: dict {(layer, head): np.array [N]}
    """
    if not contributions_per_sample:
        return {}

    # Get all head keys from first sample
    all_keys = list(contributions_per_sample[0].keys())
    contributions = {key: [] for key in all_keys}

    for sample_contribs in contributions_per_sample:
        for key in all_keys:
            contributions[key].append(sample_contribs.get(key, 0.0))

    return {key: np.array(vals) for key, vals in contributions.items()}


def select_heads(contributions, positive_mask, top_k=5):
    """
    Select top-k attention heads based on positive/negative sample contribution
    difference.

    Selection criteria:
      1. mean_pos > 0 AND mean_neg < 0 — the head boosts the correct answer
         token when the model is correct, and suppresses it when hallucinating.
         This directional signal is more discriminative than magnitude alone.
      2. Rank by |mean_pos - mean_neg| descending, take top_k.

    Fallback: if fewer than top_k heads meet criterion 1, relax to any head
    where mean_pos and mean_neg have opposite signs.

    Args:
        contributions: dict {(layer, head): np.array [B]}
        positive_mask: np.array [B] of bool — True = correct sample
        top_k: number of heads to select

    Returns:
        selected_heads: list of (layer, head)
        head_stats: dict with stats for all qualifying heads
    """
    head_stats = {}
    for key, contrib in contributions.items():
        mean_pos = contrib[positive_mask].mean()
        mean_neg = contrib[~positive_mask].mean()
        diff = mean_pos - mean_neg

        if mean_pos > 0 and mean_neg < 0:
            head_stats[key] = {
                'mean_pos': float(mean_pos),
                'mean_neg': float(mean_neg),
                'diff': float(diff),
                'abs_diff': float(abs(diff)),
            }

    # Strict selection: positive contribution for correct, negative for hallucination
    if len(head_stats) >= top_k:
        sorted_heads = sorted(head_stats.keys(),
                              key=lambda k: head_stats[k]['abs_diff'],
                              reverse=True)
        selected = sorted_heads[:top_k]
        print(f"Selected {len(selected)} heads (strict: mean_pos>0, mean_neg<0)")
    else:
        # Fallback: any head with opposite-sign contributions
        print(f"Only {len(head_stats)} heads meet strict criterion, relaxing...")
        for key, contrib in contributions.items():
            if key in head_stats:
                continue
            mean_pos = contrib[positive_mask].mean()
            mean_neg = contrib[~positive_mask].mean()
            if mean_pos * mean_neg < 0:  # opposite signs
                head_stats[key] = {
                    'mean_pos': float(mean_pos),
                    'mean_neg': float(mean_neg),
                    'diff': float(mean_pos - mean_neg),
                    'abs_diff': float(abs(mean_pos - mean_neg)),
                }

        sorted_heads = sorted(head_stats.keys(),
                              key=lambda k: head_stats[k]['abs_diff'],
                              reverse=True)
        selected = sorted_heads[:top_k]
        print(f"Selected {len(selected)} heads (relaxed: opposite signs)")

    return selected, head_stats


# ---------------------------------------------------------------------------
# Classifier-based head selection (new approach)
# ---------------------------------------------------------------------------

def build_head_probe(input_dim=1536):
    """
    Build a 3-layer feedforward binary classifier for a single attention head.
    Architecture: Dense(256, relu) → Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_head_classifiers(features_train, labels_train,
                            num_layers=28, num_heads=12, epochs=50, batch_size=32):
    """
    Train a binary classifier for each of the 336 attention heads.

    Args:
        features_train: dict {(layer, head): np.array [N_train, d_model]}
        labels_train: np.array [N_train]
        num_layers: number of transformer layers
        num_heads: number of attention heads per layer
        epochs: max training epochs (with early stopping)
        batch_size: training batch size

    Returns:
        all_classifiers: dict {(layer, head): (model, scaler)}
    """
    all_classifiers = {}

    total = num_layers * num_heads
    with tqdm(total=total, desc="Training classifiers") as pbar:
        for layer in range(num_layers):
            for head in range(num_heads):
                key = (layer, head)
                X = features_train[key]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = build_head_probe(input_dim=X.shape[1])
                model.fit(X_scaled, labels_train,
                          epochs=epochs, batch_size=batch_size, verbose=0,
                          callbacks=[EarlyStopping(patience=5,
                                                   restore_best_weights=True)])

                all_classifiers[key] = (model, scaler)
                pbar.update(1)

    return all_classifiers


def select_heads_by_auc(all_classifiers, features_val, labels_val, top_k=5):
    """
    Evaluate each head's classifier on the validation set and select top-k by AUC.

    Args:
        all_classifiers: dict {(layer, head): (model, scaler)}
        features_val: dict {(layer, head): np.array [N_val, d_model]}
        labels_val: np.array [N_val]
        top_k: number of heads to select

    Returns:
        selected_heads: list of (layer, head) — top-k by validation AUC
        head_aucs: dict {(layer, head): float} — AUC for all heads
    """
    head_aucs = {}

    for key, (model, scaler) in all_classifiers.items():
        X_val_scaled = scaler.transform(features_val[key])
        val_probs = model.predict(X_val_scaled, verbose=0).flatten()
        auc = roc_auc_score(labels_val, val_probs)
        head_aucs[key] = auc

    sorted_heads = sorted(head_aucs.keys(),
                          key=lambda k: head_aucs[k],
                          reverse=True)
    selected_heads = sorted_heads[:top_k]

    return selected_heads, head_aucs


def process_samples_for_all_heads(model, tokenizer, samples_df):
    """
    Extract influence vector features at answer last token position for ALL
    336 attention heads (not just selected ones). Used for training classifiers.

    Args:
        model: the loaded model
        tokenizer: the tokenizer
        samples_df: DataFrame with 'question', 'generated_answer', 'label' columns

    Returns:
        features: dict {(layer, head): np.array [N, d_model]}
        labels: np.array [N]
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    d_model = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device

    all_features = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}
    all_labels = []

    for idx in tqdm(range(len(samples_df)), desc="Extracting all-head features"):
        row = samples_df.iloc[idx]
        question = row['question']
        generated_answer = row['generated_answer']
        label = row['label']

        if not generated_answer.strip():
            continue

        # Build full text (prompt + generated answer)
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

        full_text = prompt_text + generated_answer
        full_inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = full_inputs["input_ids"].to(device)
        total_length = input_ids.shape[1]

        answer_last_pos = total_length - 1
        if answer_last_pos < prompt_length:
            continue

        # Register hooks on all layers' o_proj
        o_proj_inputs = {}
        hook_handles = []

        for layer_idx in range(num_layers):
            o_proj_module = model.model.layers[layer_idx].self_attn.o_proj

            def make_hook(l_idx):
                def hook_fn(module, input, output):
                    o_proj_inputs[l_idx] = input[0][0].detach().float().cpu().numpy()
                return hook_fn

            handle = o_proj_module.register_forward_hook(make_hook(layer_idx))
            hook_handles.append(handle)

        # Forward pass
        with torch.no_grad():
            model(input_ids)

        # Compute influence vectors for ALL heads
        for layer in range(num_layers):
            if layer not in o_proj_inputs:
                for head in range(num_heads):
                    all_features[(layer, head)].append(np.zeros(d_model))
                continue

            attn_out = o_proj_inputs[layer]  # [T, num_heads * head_dim]
            W_O = model.model.layers[layer].self_attn.o_proj.weight.detach()
            W_O_np = W_O.float().cpu().numpy()

            for head in range(num_heads):
                head_out = attn_out[answer_last_pos,
                                    head * head_dim:(head + 1) * head_dim]
                W_O_h = W_O_np[:, head * head_dim:(head + 1) * head_dim]
                delta_h = head_out @ W_O_h.T  # [d_model]
                all_features[(layer, head)].append(delta_h)

        # Clean up hooks
        for h in hook_handles:
            h.remove()

        all_labels.append(label)

    features = {key: np.stack(vals, axis=0) for key, vals in all_features.items()}
    labels = np.array(all_labels)
    return features, labels


def plot_auc_heatmap(head_aucs, num_layers, num_heads, save_path):
    """Plot a heatmap of per-head classifier AUC scores."""
    auc_matrix = np.zeros((num_layers, num_heads))
    for (layer, head), auc_val in head_aucs.items():
        auc_matrix[layer, head] = auc_val

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(auc_matrix, ax=ax, cmap='YlOrRd',
                vmin=0.5, vmax=1.0,
                xticklabels=[f"H{h}" for h in range(num_heads)],
                yticklabels=[f"L{l}" for l in range(num_layers)],
                annot=True, fmt='.3f', linewidths=0.5)
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title("Per-Head Classifier AUC on Validation Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"AUC heatmap saved to {save_path}")


def plot_contribution_heatmap(head_stats, num_layers, num_heads, save_path):
    """Plot a heatmap of positive-negative contribution differences."""
    diff_matrix = np.zeros((num_layers, num_heads))
    for (layer, head), stats in head_stats.items():
        diff_matrix[layer, head] = stats['diff']

    fig, ax = plt.subplots(figsize=(14, 10))
    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    sns.heatmap(diff_matrix, ax=ax, cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax,
                xticklabels=[f"H{h}" for h in range(num_heads)],
                yticklabels=[f"L{l}" for l in range(num_layers)])
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title("Positive − Negative Logit Contribution per Head")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {save_path}")


def process_samples_for_contributions(model, tokenizer, samples_df,
                                       batch_size=4):
    """
    Pass 1: Process samples to compute per-head logit contributions at the
    answer's last token position. Uses per-sample hooks that compute
    contributions directly without storing full activations.

    Returns:
        contributions_per_sample: list[dict {(layer, head): float}]
        all_labels: list[int]
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers

    # Pre-compute W_U on CPU for efficient hook computation
    W_U = model.lm_head.weight.detach().float().cpu().numpy().T  # [d_model, vocab]

    contributions_per_sample = []
    all_labels = []

    for idx in tqdm(range(len(samples_df)),
                    desc="Pass 1: computing contributions"):
        row = samples_df.iloc[idx]
        question = row['question']
        generated_answer = row['generated_answer']
        label = row['label']

        if not generated_answer.strip():
            continue

        # Build prompt in chat format
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

        full_text = prompt_text + generated_answer
        full_inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = full_inputs["input_ids"].to(next(model.parameters()).device)
        total_length = input_ids.shape[1]

        answer_last_pos = total_length - 1
        if answer_last_pos < prompt_length:
            continue

        answer_last_token_id = input_ids[0, answer_last_pos].item()

        # Register hooks that compute contributions at all positions
        # We'll extract the answer position contribution after forward pass
        sample_deltas = {}  # {(layer, head): np.array [T, d_model]}
        hook_handles = []

        for layer_idx in range(num_layers):
            o_proj_module = model.model.layers[layer_idx].self_attn.o_proj

            def make_hook(l_idx):
                def hook_fn(module, input, output):
                    attn_input = input[0][0]  # [T, num_heads * head_dim]
                    W_O = module.weight.detach()  # [d_model, num_heads * head_dim]
                    for h in range(num_heads):
                        head_out = attn_input[:, h * head_dim:(h + 1) * head_dim]
                        W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]
                        delta = torch.matmul(head_out.float(), W_O_h.float().T)
                        sample_deltas[(l_idx, h)] = delta.cpu().numpy()
                return hook_fn

            handle = o_proj_module.register_forward_hook(make_hook(layer_idx))
            hook_handles.append(handle)

        # Forward pass
        with torch.no_grad():
            model(input_ids)

        # Compute scalar contributions at answer position
        target_vec = W_U[:, answer_last_token_id]  # [d_model]
        sample_contribs = {}
        for (l, h), delta in sample_deltas.items():
            sample_contribs[(l, h)] = float(delta[answer_last_pos] @ target_vec)

        # Clean up hooks
        for h in hook_handles:
            h.remove()
        del sample_deltas

        contributions_per_sample.append(sample_contribs)
        all_labels.append(label)

    return contributions_per_sample, all_labels


def process_samples_for_selected_features(model, tokenizer, samples_df,
                                           selected_heads, batch_size=4):
    """
    Pass 2: Extract influence vector features at answer last token position
    only for the selected heads. Uses hooks only on layers containing
    selected heads.

    Returns:
        features: dict {head_key: np.array [N, d_model]}
        labels: np.array [N]
    """
    d_model = model.config.hidden_size
    head_keys = [f"L{l}_H{h}" for l, h in selected_heads]
    all_features = {key: [] for key in head_keys}
    all_labels = []

    layers_needed = list(set(l for l, h in selected_heads))

    for idx in tqdm(range(len(samples_df)),
                    desc="Pass 2: extracting selected head features"):
        row = samples_df.iloc[idx]
        question = row['question']
        generated_answer = row['generated_answer']
        label = row['label']

        if not generated_answer.strip():
            continue

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

        full_text = prompt_text + generated_answer
        full_inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = full_inputs["input_ids"].to(next(model.parameters()).device)
        total_length = input_ids.shape[1]

        answer_last_pos = total_length - 1
        if answer_last_pos < prompt_length:
            continue

        # Register hooks only on needed layers
        o_proj_inputs = {}
        hook_handles = []

        for layer_idx in layers_needed:
            o_proj_module = model.model.layers[layer_idx].self_attn.o_proj

            def make_hook(l_idx):
                def hook_fn(module, input, output):
                    o_proj_inputs[l_idx] = input[0][0].detach().float().cpu().numpy()
                return hook_fn

            handle = o_proj_module.register_forward_hook(make_hook(layer_idx))
            hook_handles.append(handle)

        # Forward pass
        with torch.no_grad():
            model(input_ids)

        # Compute influence vectors for selected heads at answer position
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads

        for layer, head in selected_heads:
            key = f"L{layer}_H{head}"
            if layer not in o_proj_inputs:
                all_features[key].append(np.zeros(d_model))
                continue

            attn_out = o_proj_inputs[layer]  # [T, num_heads * head_dim]
            W_O = model.model.layers[layer].self_attn.o_proj.weight.detach()
            W_O_np = W_O.float().cpu().numpy()

            head_out = attn_out[answer_last_pos,
                                head * head_dim:(head + 1) * head_dim]  # [head_dim]
            W_O_h = W_O_np[:, head * head_dim:(head + 1) * head_dim]  # [d_model, head_dim]
            delta_h = head_out @ W_O_h.T  # [d_model]
            all_features[key].append(delta_h)

        # Clean up hooks
        for h in hook_handles:
            h.remove()

        all_labels.append(label)

    features = {key: np.stack(vals, axis=0) for key, vals in all_features.items()}
    labels = np.array(all_labels)
    return features, labels


def run_attribution(config_path="config.json",
                    set_a_path=None,
                    set_b_path=None,
                    output_dir=None,
                    top_k=5,
                    epochs=50,
                    batch_size=32):
    """
    Full pipeline: train 336 per-head classifiers on set_a, evaluate AUC on
    set_b, select top-k heads, and save classifiers + features.

    Args:
        config_path: path to config.json
        set_a_path: path to set_a CSV (training)
        set_b_path: path to set_b CSV (validation)
        output_dir: output directory
        top_k: number of heads to select
        epochs: max training epochs per classifier
        batch_size: training batch size
    """
    config = load_config(config_path)

    model_path = config.get("model_path", "model")
    dtype_str = config.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device_map = config.get("device_map", "auto")
    processed_path = Path(config.get("processed_dataset_path", "processed_datasets"))

    if set_a_path is None:
        set_a_path = processed_path / "triviaqa_set_a.csv"
    if set_b_path is None:
        set_b_path = processed_path / "triviaqa_set_b.csv"
    if output_dir is None:
        output_dir = processed_path
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = init_model(model_path, dtype=dtype, device_map=device_map)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    # Load set_a (training) and set_b (validation)
    df_a = pd.read_csv(set_a_path)
    df_b = pd.read_csv(set_b_path)
    print(f"Set A (training):   {len(df_a)} samples, "
          f"{df_a['label'].sum()} correct, {(1 - df_a['label']).sum()} hallucination")
    print(f"Set B (validation): {len(df_b)} samples, "
          f"{df_b['label'].sum()} correct, {(1 - df_b['label']).sum()} hallucination")

    # Step 1: Extract influence vector features for ALL 336 heads
    print(f"\n{'='*60}")
    print(f"Step 1: Extracting features for all {num_layers * num_heads} heads")
    print(f"{'='*60}")

    print("\n--- Set A (training) ---")
    features_a, labels_a = process_samples_for_all_heads(model, tokenizer, df_a)

    print("\n--- Set B (validation) ---")
    features_b, labels_b = process_samples_for_all_heads(model, tokenizer, df_b)

    # Step 2: Train 336 classifiers on set_a
    print(f"\n{'='*60}")
    print(f"Step 2: Training {num_layers * num_heads} classifiers on set A")
    print(f"{'='*60}")

    all_classifiers = train_head_classifiers(
        features_a, labels_a,
        num_layers=num_layers, num_heads=num_heads,
        epochs=epochs, batch_size=batch_size
    )

    # Step 3: Evaluate on set_b and select top-k heads
    print(f"\n{'='*60}")
    print(f"Step 3: Selecting top-{top_k} heads by validation AUC")
    print(f"{'='*60}")

    selected_heads, head_aucs = select_heads_by_auc(
        all_classifiers, features_b, labels_b, top_k=top_k
    )

    print(f"\nTop-{top_k} selected heads:")
    for i, (layer, head) in enumerate(selected_heads):
        auc_val = head_aucs[(layer, head)]
        print(f"  {i+1}. Layer {layer}, Head {head}: AUC = {auc_val:.4f}")

    # Step 4: Save results
    print(f"\n{'='*60}")
    print("Step 4: Saving results")
    print(f"{'='*60}")

    # 4a. Save head AUC rankings
    auc_rows = []
    for (layer, head), auc_val in head_aucs.items():
        auc_rows.append({
            'layer': layer,
            'head': head,
            'auc': auc_val,
        })
    auc_df = pd.DataFrame(auc_rows)
    auc_df = auc_df.sort_values('auc', ascending=False)
    auc_file = output_dir / "head_aucs.csv"
    auc_df.to_csv(auc_file, index=False)
    print(f"Saved head AUC rankings to {auc_file}")

    # 4b. Save selected heads info
    top_heads_info = {
        "selected_heads": [(int(l), int(h)) for l, h in selected_heads],
        "head_details": {
            f"L{l}_H{h}": {"auc": head_aucs[(l, h)]}
            for l, h in selected_heads
        },
        "selection_method": "classifier_auc",
        "train_samples": len(labels_a),
        "val_samples": len(labels_b),
    }
    top_heads_file = output_dir / "top5_heads.json"
    with open(top_heads_file, 'w') as f:
        json.dump(top_heads_info, f, indent=2)
    print(f"Saved selected heads to {top_heads_file}")

    # 4c. Save selected head classifiers
    classifiers_dir = output_dir / "head_classifiers"
    classifiers_dir.mkdir(exist_ok=True)
    for layer, head in selected_heads:
        key = (layer, head)
        model_cls, scaler = all_classifiers[key]
        model_cls.save(classifiers_dir / f"L{layer}_H{head}_classifier.h5")
        np.save(classifiers_dir / f"L{layer}_H{head}_scaler_mean.npy",
                scaler.mean_)
        np.save(classifiers_dir / f"L{layer}_H{head}_scaler_scale.npy",
                scaler.scale_)
    print(f"Saved {top_k} classifiers to {classifiers_dir}")

    # 4d. Save features for downstream use (HeadProbe)
    np.savez(output_dir / "set_a_all_features.npz",
             labels=labels_a,
             **{f"L{l}_H{h}": features_a[(l, h)]
                for l, h in selected_heads})
    np.savez(output_dir / "set_b_selected_features.npz",
             labels=labels_b,
             **{f"L{l}_H{h}": features_b[(l, h)]
                for l, h in selected_heads})
    print(f"Saved features for selected heads")

    # Step 5: Visualization
    print(f"\n{'='*60}")
    print("Step 5: Generating visualizations")
    print(f"{'='*60}")

    heatmap_path = output_dir / "fig_head_auc_heatmap.png"
    plot_auc_heatmap(head_aucs, num_layers, num_heads, heatmap_path)

    # Clean up model
    del model, tokenizer
    torch.cuda.empty_cache()

    print("\nLogit attribution complete!")
    return selected_heads, head_aucs, all_classifiers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attention head logit attribution — classifier-based selection"
    )
    parser.add_argument("--config", default="config.json",
                        help="Path to config file")
    parser.add_argument("--set_a_path", default=None,
                        help="Path to set_a CSV (training)")
    parser.add_argument("--set_b_path", default=None,
                        help="Path to set_b CSV (validation)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top heads to select")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs per classifier")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    args = parser.parse_args()

    run_attribution(
        config_path=args.config,
        set_a_path=args.set_a_path,
        set_b_path=args.set_b_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
