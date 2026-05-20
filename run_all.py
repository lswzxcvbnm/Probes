"""
一体化运行脚本：LLaMA-2-7B 幻觉检测完整流程
- 4-bit 量化加载模型（适配 8GB 显存）
- 生成隐藏层嵌入
- 训练监督式探针
- 运行 PPL 检测
- 汇总指标
"""
import sys, os, json, subprocess, time
from pathlib import Path

BASE_DIR = Path(r"d:\code\natural_language_final\final_project\Probes")
MODEL_PATH = r"d:\code\natural_language_final\final_project\models\Llama-2-7b"
MODEL_ALIAS = "llama2_7b"

DATASETS = ["animals", "cities", "companies", "elements", "inventions", "facts"]
LAYERS = [16, 20, 24, 28, 32]
BATCH_SIZE = 4
DEFAULT_CONFIG = {
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
    "gen_predictions_dataset": f"embeddings_cities{MODEL_ALIAS}_16_rmv_period",
    "gen_predictions_layer": -4,
    "suffix_list": [f"{d}_rp" for d in DATASETS]
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def step1_generate_embeddings():
    """步骤1: 生成嵌入"""
    log("=== 步骤1: 生成 LLaMA-2-7B 嵌入向量 ===")
    os.chdir(str(BASE_DIR))

    config_path = BASE_DIR / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        log("已写入 config.json")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from LLaMa_generate_embeddings import process_batch, load_data, save_data
    from tqdm import tqdm
    import pandas as pd
    import numpy as np

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
    log("模型加载完成 (4-bit量化)")

    dataset_path = Path("datasets")
    output_path = Path("processed_datasets")
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        log(f"处理数据集: {dataset_name}")
        df = load_data(dataset_path, dataset_name, true_false=True)
        if df is None:
            log(f"  跳过: 找不到 {dataset_name}")
            continue

        texts = df["statement"].astype(str).tolist()
        should_remove_period = True

        for layer in LAYERS:
            log(f"  层 {layer}...")
            df_out = df.copy()
            df_out["embeddings"] = pd.Series(dtype="object")

            num_batches = len(texts) // BATCH_SIZE + (len(texts) % BATCH_SIZE != 0)
            for batch_num in tqdm(range(num_batches), desc=f"  {dataset_name}/L{layer}"):
                start = batch_num * BATCH_SIZE
                end = min(start + BATCH_SIZE, len(texts))
                batch_texts = texts[start:end]

                batch_embs = process_batch(
                    batch_texts, model, tokenizer, [layer], should_remove_period
                )

                for i, idx in enumerate(range(start, end)):
                    df_out.at[idx, "embeddings"] = batch_embs[layer][i]

                torch.cuda.empty_cache()

            save_data(df_out, output_path, dataset_name, MODEL_ALIAS, layer,
                      should_remove_period)
            log(f"  层 {layer} 完成 -> embeddings_{dataset_name}{MODEL_ALIAS}_{layer}_rmv_period.csv")

    log("=== 步骤1 完成 ===")
    del model
    torch.cuda.empty_cache()


def step2_train_probes():
    """步骤2: 训练监督式探针"""
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
    import os as _os

    config = load_config("config.json")

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
        overall_res = []

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
                model_path_val = probes_path / f"{model_name}_{layer_num}_{dataset_names[count]}{suffix}.h5"
                best_model.save(str(model_path_val))

        avg_res = print_results(results, dataset_names, repeat_each, layer_num)
        overall_res.extend(avg_res)
        metrics_rows.extend(collect_metrics(results, dataset_names, repeat_each, layer_num))

    metrics_path = "processed_datasets/supervised_probe_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    log(f"指标已保存: {metrics_path}")

    for row in metrics_rows:
        log(f"  dataset={row['dataset']} layer={row['layer']} "
            f"acc={row['avg_accuracy']:.4f} auc={row['avg_auc']:.4f}")

    log("=== 步骤2 完成 ===")


def step3_ppl_detection():
    """步骤3: PPL 困惑度检测"""
    log("=== 步骤3: PPL 检测 ===")
    os.chdir(str(BASE_DIR))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import pandas as pd
    import numpy as np
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
        return nll.detach().cpu().numpy()

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


def main():
    log("=" * 60)
    log("LLaMA-2-7B 幻觉检测 - 完整实验流程")
    log("模型: 4-bit 量化 | GPU: RTX 4060 (8GB)")
    log("=" * 60)

    try:
        step1_generate_embeddings()
    except Exception as e:
        log(f"步骤1 错误: {e}")
        import traceback; traceback.print_exc()

    try:
        step2_train_probes()
    except Exception as e:
        log(f"步骤2 错误: {e}")
        import traceback; traceback.print_exc()

    try:
        step3_ppl_detection()
    except Exception as e:
        log(f"步骤3 错误: {e}")
        import traceback; traceback.print_exc()

    log("=" * 60)
    log("全部流程完成！查看 processed_datasets/ 目录获取结果")
    log("=" * 60)


if __name__ == "__main__":
    main()
