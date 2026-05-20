"""只运行步骤2(训练探针)和步骤3(PPL检测)"""
import sys, os, json, time
from pathlib import Path

BASE_DIR = Path(r"d:\code\natural_language_final\final_project\Probes")
MODEL_PATH = r"d:\code\natural_language_final\final_project\models\Llama-2-7b"
MODEL_ALIAS = "llama2_7b"
DATASETS = ["animals", "cities", "companies", "elements", "inventions", "facts"]
LAYERS = [16, 20, 24, 28, 32]
BATCH_SIZE = 4

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Write config needed by TrainProbes
config = {
    "model": MODEL_ALIAS,
    "remove_period": True,
    "test_first_only": False,
    "save_probes": True,
    "repeat_each": 3,
    "probes_dir": "probes",
    "layers_to_use": LAYERS,
    "layer": -4,
    "list_of_datasets": DATASETS,
    "dataset_path": "datasets",
    "processed_dataset_path": "processed_datasets",
    "true_false": True,
    "batch_size": BATCH_SIZE,
}
with open(BASE_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=4)

log("=== 步骤2: 训练探针 ===")
os.chdir(str(BASE_DIR))

from TrainProbes import (
    load_config, load_datasets, prepare_datasets,
    correct_str, define_model, train_model,
    evaluate_model, find_optimal_threshold,
    compute_roc_curve, print_results, collect_metrics
)
from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np

model_name = MODEL_ALIAS
should_remove_period = True
dataset_names = DATASETS
test_first_only = False
save_probes = True
repeat_each = 3
input_path = Path("processed_datasets")
probes_path = Path("probes")
probes_path.mkdir(parents=True, exist_ok=True)

metrics_rows = []
for idx in range(len(LAYERS)):
    layer_num = LAYERS[idx]
    log(f"处理层: {layer_num}")

    datasets, dataset_paths = load_datasets(
        dataset_names, LAYERS, should_remove_period,
        input_path, model_name, idx
    )
    train_datasets, test_datasets = prepare_datasets(
        datasets, dataset_names, test_first_only
    )

    results = []
    for count, (test_ds, train_ds, test_ds_path) in enumerate(
        zip(test_datasets, train_datasets, dataset_paths)
    ):
        train_embs = np.array([
            np.fromstring(correct_str(e), sep=",")
            for e in train_ds["embeddings"].tolist()
        ])
        train_labels = train_ds["label"]
        test_embs = np.array([
            np.fromstring(correct_str(e), sep=",")
            for e in test_ds["embeddings"].tolist()
        ])
        test_labels = test_ds["label"]

        best_accuracy = 0
        best_model = None
        all_probs = []

        for i in range(repeat_each):
            model = define_model(train_embs.shape[1])
            model = train_model(model, train_embs, train_labels)

            try:
                thr = find_optimal_threshold(train_embs, train_labels, model)
            except:
                thr = 0.5
            loss, accuracy = evaluate_model(model, test_embs, test_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

            probs = model.predict(test_embs)
            all_probs.append(deepcopy(probs))
            roc_auc, _, _ = compute_roc_curve(test_labels, probs)
            test_acc = ((probs > thr).astype(int).flatten() == test_labels).mean()

            results.append((dataset_names[count], i, accuracy, roc_auc, thr, test_acc))

        if save_probes and best_model:
            suffix = "_rp" if should_remove_period else ""
            mp = probes_path / f"{model_name}_{layer_num}_{dataset_names[count]}{suffix}.h5"
            best_model.save(str(mp))
            log(f"  保存探针: {mp.name} (acc={best_accuracy:.4f})")

    print_results(results, dataset_names, repeat_each, layer_num)
    metrics_rows.extend(collect_metrics(results, dataset_names, repeat_each, layer_num))

metrics_path = "processed_datasets/supervised_probe_metrics.csv"
pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
log(f"探针指标已保存: {metrics_path}")

for row in metrics_rows:
    log(f"  {row['dataset']}/L{row['layer']}: acc={row['avg_accuracy']:.4f} auc={row['avg_auc']:.4f}")

log("=== 步骤2 完成 ===")


log("=== 步骤3: PPL 检测 ===")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def compute_nll_batch(texts, tokenizer, model, device):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
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
    nll = -gathered * shift_mask
    token_counts = shift_mask.sum(dim=1).clamp(min=1)
    nll = nll.sum(dim=1) / token_counts
    return nll.detach().cpu().float().numpy()

def find_best_threshold(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_thr, best_acc = thresholds[0], -1.0
    for thr in thresholds:
        acc = accuracy_score(labels, scores >= thr)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc

log(f"加载模型: {MODEL_PATH} (4-bit量化)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()
device = "cuda"
log("模型加载完成")

dataset_path = Path("datasets")
output_path = Path("processed_datasets")
output_path.mkdir(parents=True, exist_ok=True)

summary_rows = []
for dataset_name in DATASETS:
    log(f"PPL 处理: {dataset_name}")
    df = pd.read_csv(dataset_path / f"{dataset_name}_true_false.csv")
    texts = df["statement"].astype(str).tolist()

    all_nll = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        nll_vals = compute_nll_batch(batch, tokenizer, model, device)
        all_nll.extend(nll_vals.tolist())

    df["nll"] = np.array(all_nll)

    if "label" in df.columns:
        labels = (1 - df["label"].astype(int)).to_numpy()
        scores = df["nll"].to_numpy()
        finite = np.isfinite(scores)
        labels_f, scores_f = labels[finite], scores[finite]

        if len(np.unique(labels_f)) >= 2:
            thr, acc = find_best_threshold(scores_f, labels_f)
            try:
                auc_val = roc_auc_score(labels_f, scores_f)
            except:
                auc_val = None
            log(f"  {dataset_name}: acc={acc:.4f} auc={auc_val} thr={thr:.4f}")
            summary_rows.append({
                "dataset": dataset_name,
                "accuracy": acc,
                "auc": auc_val,
                "best_threshold": thr,
            })

    df.to_csv(output_path / f"{dataset_name}_true_false_ppl_predictions.csv", index=False)

summary_df = pd.DataFrame(summary_rows)
ppl_metrics_path = output_path / "ppl_metrics.csv"
summary_df.to_csv(ppl_metrics_path, index=False)
log(f"PPL 指标已保存: {ppl_metrics_path}")
log("=== 步骤3 完成 ===")

log("=" * 60)
log("全部完成！")
log("=" * 60)
