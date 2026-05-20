"""PPL 检测 - LLaMA-2-7B 4-bit量化"""
import time, os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

BASE_DIR = Path(r"d:\code\natural_language_final\final_project\Probes")
MODEL_PATH = r"d:\code\natural_language_final\final_project\models\Llama-2-7b"
DATASETS = ["animals", "cities", "companies", "elements", "inventions", "facts"]
BATCH = 4
os.chdir(str(BASE_DIR))

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def compute_nll(texts, tok, model, dev):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
    ids = enc["input_ids"].to(dev)
    mask = enc["attention_mask"].to(dev)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    sl = out.logits[:, :-1, :]
    st = ids[:, 1:]
    sm = mask[:, 1:]
    lp = torch.log_softmax(sl, dim=-1)
    g = lp.gather(dim=-1, index=st.unsqueeze(-1)).squeeze(-1)
    nll = -g * sm
    tc = sm.sum(dim=1).clamp(min=1)
    return (nll.sum(dim=1) / tc).detach().cpu().float().numpy()

def best_thr(scores, labels):
    fpr, tpr, thrs = roc_curve(labels, scores)
    bt, ba = thrs[0], -1.0
    for t in thrs:
        a = accuracy_score(labels, scores >= t)
        if a > ba:
            ba, bt = a, t
    return bt, ba

log("加载 4-bit 量化模型...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bcfg, device_map="auto")
model.eval()
dev = "cuda"
log("模型就绪")

out = Path("processed_datasets")
out.mkdir(parents=True, exist_ok=True)
rows = []

for ds in DATASETS:
    log(f"PPL: {ds}")
    df = pd.read_csv(Path("datasets") / f"{ds}_true_false.csv")
    texts = df["statement"].astype(str).tolist()
    nlls = []
    for s in range(0, len(texts), BATCH):
        nlls.extend(compute_nll(texts[s:s + BATCH], tok, model, dev))
    df["nll"] = np.array(nlls)

    if "label" in df.columns:
        labels = (1 - df["label"].astype(int)).to_numpy()
        scores = df["nll"].to_numpy()
        fin = np.isfinite(scores)
        lf, sf = labels[fin], scores[fin]
        if len(np.unique(lf)) >= 2:
            t, a = best_thr(sf, lf)
            try:
                auc = roc_auc_score(lf, sf)
            except:
                auc = None
            log(f"  {ds}: acc={a:.4f} auc={auc} thr={t:.4f}")
            rows.append({"dataset": ds, "accuracy": a, "auc": auc, "best_threshold": t})
    df.to_csv(out / f"{ds}_true_false_ppl_predictions.csv", index=False)

pd.DataFrame(rows).to_csv(out / "ppl_metrics.csv", index=False)
log("PPL 完成: ppl_metrics.csv")
