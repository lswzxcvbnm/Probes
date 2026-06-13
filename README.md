# 基于大语言模型内部状态的幻觉检测

本项目探索利用 LLM 内部状态进行幻觉检测的三种方法，基于 Qwen2-1.5B（28 层 Transformer，每层 12 个注意力头，hidden_size=1536）。

## 项目结构

```
├── config.json                      # 主配置文件（模型路径、数据集、实验参数）
├── plan.md                          # 阶段三详细研究计划
├── Milestone.md                     # 中期进度报告
│
├── # 阶段一/二：基线实验
├── Generate_PPL_predictions.py      # PPL 困惑度幻觉检测
├── Qwen2_SAPLMA.py                  # SAPLMA 隐藏层探针实验
├── Qwen2_TokenPosition.py           # Token 位置分析实验
│
├── # 阶段三：注意力头分类器筛选与投票
├── Qwen2_TriviaQA_generate.py       # TriviaQA 答案生成与自动标注
├── Qwen2_HeadsAttribution.py        # 336 个逐头分类器训练与 Top-k 筛选
├── Qwen2_HeadProbe.py               # 探针重训练与投票集成评估
│
├── # 通用工具
├── TrainProbes.py                   # 有监督探针训练（OPT/LLaMA）
├── Generate_Embeddings.py           # OPT 嵌入提取
├── LLaMa_generate_embeddings.py     # LLaMA 嵌入提取
├── Train_CCSProbe.py                # CCS 无监督探针训练
├── Generate_CCS_predictions.py      # CCS 探针预测
├── plot_charts.py                   # 可视化图表生成
│
├── datasets/                        # 原始数据集（true/false CSV）
└── processed_datasets/              # 实验结果、指标、图表
```

## 环境配置

```bash
pip install torch transformers tensorflow scikit-learn pandas numpy matplotlib seaborn tqdm
```

在 `config.json` 中配置模型路径：

```json
{
    "model_path": "/path/to/Qwen2-1.5B",
    "model_alias": "qwen2_1.5b",
    "dtype": "bfloat16",
    "device_map": "auto"
}
```

---

## 阶段一：PPL 与 SAPLMA 基线

### 1.1 PPL 困惑度幻觉检测

基于语言模型困惑度的无参数方法：对 true/false 数据集中的每条陈述计算负对数似然（NLL），以此作为幻觉分数，通过 ROC 曲线寻优阈值进行分类。

```bash
python Generate_PPL_predictions.py \
  --model Qwen2-1.5B \
  --dataset_names animals cities companies elements inventions facts \
  --true_false \
  --config config.json
```

**输出**：`processed_datasets/{dataset}_ppl_predictions.csv`，包含每条样本的 NLL 和预测结果。

### 1.2 SAPLMA 隐藏层探针

在 Qwen2-1.5B 的多个中间层提取最后一个 token 的隐藏状态，训练三层前馈网络探针（256→128→64→1），采用留一法交叉验证（5 个主题训练，1 个主题测试）。

```bash
python Qwen2_SAPLMA.py --config config.json
```

**配置**：在 `config.json` 中指定实验参数：
- `layers_to_use`：测试的层（默认 `[14, 18, 22, 26, 28]`）
- `list_of_datasets`：数据集列表（默认 6 个主题）
- `repeat_each`：每组实验随机重启次数（默认 3）
- `epochs`：训练轮数（默认 5）

**输出**：`processed_datasets/qwen2_saplma_metrics.csv`，包含每层每主题的 Accuracy 和 AUC。

---

## 阶段二：Token 位置分析

在 Layer 18 上比较不同 token 位置（第 1、第 2、倒数第 2、最后一个 token）对探针性能的影响，验证"答案末尾 token 承载最多语义信息"的假设。

```bash
python Qwen2_TokenPosition.py --config config.json
```

**输出**：`processed_datasets/qwen2_token_position_metrics.csv`，包含 4 个位置 × 6 个主题的评估指标。

---

## 阶段三：注意力头分类器筛选与投票式幻觉检测

阶段三的核心思路：为 Qwen2-1.5B 的全部 336 个注意力头（28 层 × 12 头）各训练一个二分类器，以该头的影响向量（influence vector）为特征，通过验证集 AUC 筛选 Top-5 头，最终通过投票集成进行幻觉检测。

### 步骤 1：生成 TriviaQA 答案并划分数据集

从 TriviaQA 数据集中采样问题，使用 Qwen2-1.5B 以 greedy decoding 生成答案，通过模糊匹配自动标注正确/幻觉，随机划分为三个互不重叠的集合。

```bash
python Qwen2_TriviaQA_generate.py \
  --config config.json \
  --max_samples 2000 \
  --n_a 500 --n_b 300 --n_c 200 \
  --max_new_tokens 50
```

**参数说明**：
- `--max_samples`：从 TriviaQA 中采样的最大问题数（默认 2000，足够划分 500+300+200）
- `--n_a` / `--n_b` / `--n_c`：训练集 / 验证集 / 测试集的样本数
- `--max_new_tokens`：每条答案的最大生成 token 数

**输出**：
- `processed_datasets/triviaqa_set_a.csv`（训练集，~500 条）
- `processed_datasets/triviaqa_set_b.csv`（验证集，~300 条）
- `processed_datasets/triviaqa_set_c.csv`（测试集，~200 条）

### 步骤 2：训练 336 个逐头分类器并筛选 Top-k 头

为全部 336 个注意力头各训练一个独立的二分类器（三层前馈网络：256→128→64→1），在验证集上评估 AUC，筛选出判别力最强的 Top-k 个头。

```bash
python Qwen2_HeadsAttribution.py \
  --config config.json \
  --top_k 5 \
  --epochs 50 \
  --batch_size 32
```

**参数说明**：
- `--top_k`：筛选的注意力头数量（默认 5）
- `--epochs`：每个分类器的最大训练轮数（配合 early stopping，patience=5）
- `--batch_size`：训练 batch size

**输出**：
- `processed_datasets/head_aucs.csv`：全部 336 个头的验证集 AUC 排名
- `processed_datasets/top5_heads.json`：Top-5 头的索引及 AUC
- `processed_datasets/head_classifiers/`：Top-5 头的分类器模型和 scaler
- `processed_datasets/set_a_all_features.npz`：训练集 Top-5 头特征
- `processed_datasets/set_b_selected_features.npz`：验证集 Top-5 头特征
- `processed_datasets/fig_head_auc_heatmap.png`：336 头 AUC 热力图

### 步骤 3：探针重训练与投票集成评估

在选定的 Top-k 头上重新训练探针（5 epoch × 3 次随机重启 + 最优阈值选择），在测试集上评估多种方法的端到端性能：PPL 基线、SAPLMA 基线、单头探针、求和探针、投票集成。

```bash
python Qwen2_HeadProbe.py \
  --config config.json \
  --top_k 5 \
  --repeat_each 3 \
  --epochs 5 \
  --hidden_layer 18
```

**参数说明**：
- `--top_k`：使用的注意力头数量（应与步骤 2 一致）
- `--repeat_each`：每个探针的随机重启次数
- `--epochs`：每次重启的训练轮数
- `--hidden_layer`：SAPLMA 基线使用的隐藏层（默认 18）

**输出**：
- `processed_datasets/head_probe_metrics.csv`：所有方法的评估指标（Accuracy / Precision / Recall / F1 / AUC）
- `processed_datasets/head_probes/`：重新训练的探针模型
- `processed_datasets/fig_voting_comparison.png`：方法对比柱状图

---

## 实验结果概览

### SAPLMA 各层性能（留一法平均）

| Layer | Accuracy | AUC   |
|-------|----------|-------|
| 14    | 0.719    | 0.810 |
| 18    | 0.739    | 0.834 |
| 22    | 0.724    | 0.818 |
| 26    | 0.696    | 0.790 |
| 28    | 0.656    | 0.761 |

### Top-5 注意力头（验证集 AUC）

| 排名 | 注意力头 | 验证集 AUC |
|------|----------|-----------|
| 1    | L15_H6   | 0.7823    |
| 2    | L13_H11  | 0.7793    |
| 3    | L15_H9   | 0.7781    |
| 4    | L16_H8   | 0.7727    |
| 5    | L13_H6   | 0.7606    |

### 端到端幻觉检测（TriviaQA 测试集，203 条）

| 方法              | Accuracy | Precision | Recall | F1    | AUC   |
|-------------------|----------|-----------|--------|-------|-------|
| PPL               | 0.625    | 0.889     | 0.098  | 0.176 | 0.539 |
| SAPLMA-L18        | 0.680    | 0.667     | 0.439  | 0.529 | 0.723 |
| 求和探针 (Top-5)  | **0.710**| **0.676** | 0.561  | 0.613 | **0.776** |
| 投票集成 (Top-5)  | 0.685    | 0.620     | 0.598  | 0.609 | 0.762 |

**关键发现**：仅使用 5/336 个注意力头的影响向量，求和探针（AUC 0.776）即超越 SAPLMA（AUC 0.723，使用完整 1536 维隐藏状态），同时揭示了对幻觉检测最关键的注意力头位于第 13-16 层。
