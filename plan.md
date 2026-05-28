# 基于注意力归因的直接 Logit 贡献 — 幻觉检测研究计划

## 1. 研究背景与动机

SAPLMA 方法通过提取隐藏层表征训练探针来检测幻觉，已在阶段二中验证了其有效性（Layer 18 平均 AUC 0.834）。然而，SAPLMA 将整个隐藏状态作为黑盒特征，并未揭示模型内部哪些组件对"知道答案是否正确"起到了关键作用。

本阶段的目标是**从注意力头层面解释幻觉检测的可归因性**：在 QA 生成任务上，计算每个注意力头对答案最后一个 token logit 的直接贡献，基于正确生成与幻觉生成样本的贡献模式差异筛选关键头，并为每个关键头训练独立的二分类器，通过投票机制进行幻觉检测。

### 核心思想

在 Transformer 的前向传播中，每个注意力头 $h$ 在残差流中添加一个增量向量：

$$
\Delta h = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V \cdot W_O
$$

该向量经过 unembedding 矩阵 $W_U$ 投影后，直接影响下一个 token 的 logit 分布。在 QA 生成场景中，我们关注模型生成答案时的内部机制：对于正确生成的样本，某些头应倾向于增强正确答案 token 的 logit（正贡献）；对于幻觉样本，这些头的贡献应反转或消失（负贡献）。通过对比两类样本的贡献模式，可以识别出对幻觉检测最具判别力的注意力头。

## 2. 技术方案

### 2.1 模型与数据

| 项目             | 选择       | 说明                                                     |
| ---------------- | ---------- | -------------------------------------------------------- |
| 基座模型         | Qwen2-1.5B | 与阶段一、二一致，28 层 Transformer，每层 16 个注意力头  |
| 头筛选数据集     | TriviaQA   | 采样模型生成答案，区分正确生成与幻觉样本，用于筛选关键头 |
| 探针训练数据集   | TriviaQA   | 与头筛选不重叠，用于训练 5 个单头二分类器                |
| 端到端验证数据集 | TriviaQA   | 用于最终的幻觉检测性能评估                               |

**任务定义**：不再使用 true/false 分类 Prompt，而是直接在 QA 生成任务上进行分析。

**TriviaQA 数据格式**（单条样例）：

```json
{
  "input": [
    {"role": "system", "content": "Follow the given examples and answer the question."},
    {"role": "user", "content": "Who was the man behind The Chipmunks?"}
  ],
  "ideal": ["David Seville", "david seville"]
}
```

- `input`：Chat 格式的消息列表，包含 system 和 user 两条消息
- `ideal`：可接受答案的别名列表，用于自动标注生成答案的正确性

**Prompt 模板**：直接复用数据集中的 `input` 字段作为对话输入，无需额外构造 Prompt。模型以 greedy decoding 生成答案，然后与 `ideal` 列表进行模糊匹配，划分为：

- **正样本（正确生成）**：生成答案包含在 `ideal` 列表中
- **负样本（幻觉生成）**：生成答案不在 `ideal` 列表中

### 2.2 注意力头 Logit 贡献计算（核心步骤）

#### 步骤 1：生成答案并划分正确/幻觉样本

1. 对 TriviaQA 训练集中的每个问题，使用 Qwen2-1.5B 生成答案（greedy decoding）
2. 将生成答案与标准答案（`ideal` 字段，为可接受答案别名列表）进行模糊匹配，划分为：
   - **正样本（正确生成）**：生成答案包含在 `ideal` 列表中
   - **负样本（幻觉生成）**：生成答案不在 `ideal` 列表中
3. 按比例采样，构建用于头筛选的样本集（如各取 150 条，共 300 条）

#### 步骤 2：Hook 注册与中间量提取

在 Qwen2-1.5B 的每一层注意力模块上注册 forward hook，提取每个头在 `o_proj` 前的输出。

具体而言，对于 Qwen2 的注意力模块 `model.model.layers[l].self_attn`：

- 输入：隐藏状态 $x \in \mathbb{R}^{B \times T \times d_{model}}$
- 每个头 $h$ 计算：
  - $Q_h = x W_Q^h$, $K_h = x W_K^h$, $V_h = x W_V^h$
  - 注意力权重 $A_h = \text{softmax}(Q_h K_h^T / \sqrt{d_k})$
  - 头输出 $O_h = A_h V_h$
  - **影响向量** $\Delta_h = O_h W_O^h$，其中 $W_O^h$ 是 $W_O$ 对应第 $h$ 个头的切片

> **实现方案**：hook `attn` 模块的 forward，捕获 `attn_output`（形状 `[B, T, num_heads * head_dim]`），按 `head_dim` 切分得到每个头的输出 $O_h$，再与 `o_proj` 权重矩阵的对应切片相乘得到 $\Delta_h$。

#### 步骤 3：计算每个头对答案最后一个 token 的 Logit 贡献

对于每个输入样本，**取生成答案的最后一个 token 位置**（而非整个序列的最后一个 token）：

1. 获取 unembedding 矩阵 $W_U \in \mathbb{R}^{d_{model} \times |V|}$（即 `model.lm_head.weight`）
2. 确定目标 token ID：答案最后一个 token 的实际 token ID（每个样本的目标 token 不同）
3. 对于每个注意力头 $h$（共 $28 \times 16 = 448$ 个头），计算其对答案末尾 token logit 的贡献：

$$
\text{contrib}(h, i) = \Delta_h[\text{ans\_last\_pos}_i] \cdot W_U[:, \text{ans\_last\_token}_i]
$$

其中：

- $\Delta_h[\text{ans\_last\_pos}_i] \in \mathbb{R}^{d_{model}}$ 是该头在答案最后一个 token 位置添加到残差流的向量
- $W_U[:, \text{ans\_last\_token}_i] \in \mathbb{R}^{d_{model}}$ 是该 token 的 unembedding 向量
- 点积结果为标量，表示该头对该 token logit 的直接贡献

> **为什么选择答案最后一个 token？** 答案末尾 token 通常承载了最多的语义信息，是模型"决定说什么"的关键位置。相比之下，答案中间的 token 往往是生成过程中的过渡字符。

#### 步骤 4：基于正负样本贡献模式筛选关键头

对正样本和幻觉样本分别计算每个头的平均贡献量：

$$
\bar{C}_h^{+} = \frac{1}{N^+} \sum_{i \in \text{positive}} \text{contrib}(h, i)
$$

$$
\bar{C}_h^{-} = \frac{1}{N^-} \sum_{i \in \text{negative}} \text{contrib}(h, i)
$$

**筛选标准**：选取满足以下条件的注意力头：

$$
\bar{C}_h^{+} > 0 \quad \text{且} \quad \bar{C}_h^{-} < 0
$$

即：在正确生成样本中对答案 logit 产生**正贡献**（增强正确答案），在幻觉样本中产生**负贡献**（抑制正确答案）的头。这些头的行为与"模型是否知道答案正确"高度相关。

从满足条件的头中，按**正负样本贡献差异** $|\bar{C}_h^{+} - \bar{C}_h^{-}|$ 降序排列，选取 **Top-k 个头**（$k=5$）。

> **优势**：相比绝对值筛选，此方法直接利用了标签信息，选出的头具有明确的语义——它们在模型"自信正确"和"产生幻觉"时表现截然不同，是理想的幻觉检测特征。

### 2.3 基于 Top-k 头的投票式幻觉检测

对每个 Top-k 头，单独训练一个二分类器，最终通过投票决定样本是否为幻觉。

#### 单头二分类器训练

对每个 Top-k 头 $h_j$（$j=1,\ldots,5$），独立训练一个二分类器：

1. **特征提取**：取该头在答案最后一个 token 位置的影响向量 $\Delta_{h_j}[\text{ans\_last\_pos}] \in \mathbb{R}^{d_{model}}$ 作为特征（维度 = 1536）
2. **标签**：使用生成答案的正确性标签（1=正确生成，0=幻觉生成）
3. **分类器架构**：沿用项目已有的三层前馈网络探针

   - `Dense(256, relu) → Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)`
   - 优化器：Adam，损失函数：binary cross-entropy

#### 投票机制

对于每个测试样本，5 个独立分类器各自输出一个预测概率 $p_j \in [0, 1]$：

$$
\text{vote}(\mathbf{p}) = \begin{cases} 1 \text{ (正确)} & \text{if } \sum_{j=1}^{5} \mathbb{1}[p_j > 0.5] \geq 3 \\ 0 \text{ (幻觉)} & \text{otherwise} \end{cases}
$$

即多数投票：超过半数的分类器判定为正确，则最终判定为正确；否则判定为幻觉。

> **投票的优势**：
>
> - 降低单个头的噪声影响，提高鲁棒性
> - 每个头关注不同层面的信息（如事实记忆、推理路径、置信度），投票融合多角度判断
> - 可分析单个头的独立性能与投票后的集成性能差异

### 2.4 数据划分与端到端评估

#### 数据划分

将 TriviaQA 数据划分为三个互不重叠的子集：

| 子集   | 用途       | 样本量  | 说明                                        |
| ------ | ---------- | ------- | ------------------------------------------- |
| 集合 A | 头筛选     | ~300 条 | 正负样本各半，用于计算贡献量并筛选 Top-k 头 |
| 集合 B | 探针训练   | ~500 条 | 用于训练 5 个单头二分类器                   |
| 集合 C | 端到端验证 | ~200 条 | 用于最终评估投票式幻觉检测性能              |

> 所有子集均通过 QA 生成 → 自动标注流程，得到每条样本的正确/幻觉标签。

#### 端到端评估流程

对验证集（集合 C）中的每个问答对：

1. 用 Qwen2-1.5B 生成答案（与训练时一致的 Prompt 和解码策略）
2. 自动标注答案正确性作为真实标签
3. 提取 Top-k 头在答案最后一个 token 位置的影响向量
4. 用 5 个已训练的单头二分类器分别预测
5. 通过多数投票得到最终预测
6. 计算 **Accuracy、Precision、Recall、F1、AUC**

#### 对比基线

- **PPL 方法**：在同一验证集上，用负对数似然作为幻觉分数，ROC 寻优阈值
- **SAPLMA 方法**：在同一验证集上，提取隐藏层表征训练探针
- **单头探针（无投票）**：每个头的独立性能，分析投票的增益

## 3. 关键实现细节

### 3.1 Qwen2 注意力模块结构

Qwen2-1.5B 的注意力实现位于 `modeling_qwen2.py` 中的 `Qwen2Attention` 类：

```python
class Qwen2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        self.num_heads = config.num_attention_heads  # 16
        self.head_dim = config.hidden_size // config.num_attention_heads  # 96
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
```

**提取影响向量的关键**：在 `o_proj` 之前，`attn_output` 的形状为 `[B, T, num_heads * head_dim]`，可以按 `head_dim=96` 切分得到每个头的输出，再分别与 `o_proj.weight` 的对应列相乘。

### 3.2 Hook 实现策略

```python
# 注册 hook 到每一层的 self_attn 模块
head_outputs = {}  # {layer_idx: attn_output_before_o_proj}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # Qwen2 的 self_attn.forward() 返回 (attn_output, attn_weights, past_key_value)
        # 其中 attn_output 已经过 o_proj，形状为 [B, T, d_model]
        # 我们需要 hook 内部计算，在 o_proj 之前捕获每个头的输出
        # 方案：使用自定义 wrapper 替换 forward，或 hook o_proj 的输入
        head_outputs[layer_idx] = output[0].detach()  # 暂存，后续需要调整 hook 位置
    return hook_fn

for layer_idx in range(28):
    model.model.layers[layer_idx].self_attn.register_forward_hook(make_hook(layer_idx))
```

> **注意**：Qwen2 的 `self_attn.forward()` 返回的 `output[0]` 是经过 `o_proj` 之后的结果（形状 `[B, T, d_model]`），而非 `o_proj` 之前的头输出（形状 `[B, T, num_heads * head_dim]`）。需要通过以下方式之一获取 `o_proj` 前的输出：
>
> 1. **Hook `o_proj` 的输入**：在 `self_attn.o_proj` 上注册 forward hook，其输入即为拼接后的头输出
> 2. **自定义 wrapper**：替换 `self_attn.forward`，手动分离 Q/K/V 计算和 `o_proj` 投影
>
> 此外，需要同步记录每个样本答案最后一个 token 在序列中的位置（通过 tokenizer 的 `offset_mapping` 或逐 token 追踪生成位置）。

### 3.3 影响向量计算

```python
def compute_influence_vectors(head_outputs, o_proj_weights):
    """
    计算每个头在所有位置的影响向量（o_proj 前的头输出投影到残差流）。

    Args:
        head_outputs: dict {layer_idx: [B, T, num_heads * head_dim]} — o_proj 前的头输出
        o_proj_weights: dict {layer_idx: o_proj.weight [d_model, num_heads * head_dim]}
    Returns:
        influences: dict {(layer, head): np.array [B, T, d_model]}
    """
    influences = {}
    for layer_idx, attn_out in head_outputs.items():
        W_O = o_proj_weights[layer_idx]  # [d_model, num_heads * head_dim]
        for h in range(num_heads):
            W_O_h = W_O[:, h*head_dim:(h+1)*head_dim]  # [d_model, head_dim]
            head_out = attn_out[:, :, h*head_dim:(h+1)*head_dim]  # [B, T, head_dim]
            delta_h = (head_out @ W_O_h.T).cpu().float().numpy()  # [B, T, d_model]
            influences[(layer_idx, h)] = delta_h
    return influences
```

### 3.4 Logit 贡献计算（QA 生成场景）

```python
def compute_logit_contribution_qa(influences, W_U, answer_token_ids, answer_last_positions):
    """
    计算每个头对答案最后一个 token 的 logit 贡献。

    Args:
        influences: dict {(layer, head): np.array [B, T, d_model]} — 每个头在所有位置的影响向量
        W_U: np.array [d_model, vocab_size] — unembedding 矩阵
        answer_token_ids: list[int] — 每个样本答案最后一个 token 的 ID
        answer_last_positions: list[int] — 每个样本答案最后一个 token 在序列中的位置
    Returns:
        contributions: dict {(layer, head): np.array [B]} 每个样本的标量贡献
    """
    contributions = {}
    for key, delta in influences.items():
        contrib = []
        for i in range(len(answer_token_ids)):
            pos = answer_last_positions[i]
            token_id = answer_token_ids[i]
            # 头在答案末尾位置的影响向量与目标 token unembedding 的点积
            c = delta[i, pos] @ W_U[:, token_id]
            contrib.append(c.item())
        contributions[key] = np.array(contrib)
    return contributions


def select_heads_by_contribution(contributions, positive_mask, top_k=5):
    """
    基于正负样本贡献模式筛选关键头。

    Args:
        contributions: dict {(layer, head): np.array [B]}
        positive_mask: np.array [B] — True 表示正确生成样本，False 表示幻觉样本
        top_k: 选取的头数量
    Returns:
        selected_heads: list of (layer, head) — 按正负差异排序的 Top-k 头
        head_stats: dict {(layer, head): {'mean_pos': float, 'mean_neg': float, 'diff': float}}
    """
    head_stats = {}
    for key, contrib in contributions.items():
        mean_pos = contrib[positive_mask].mean()
        mean_neg = contrib[~positive_mask].mean()
        # 筛选条件：正样本贡献为正，负样本贡献为负
        if mean_pos > 0 and mean_neg < 0:
            head_stats[key] = {
                'mean_pos': mean_pos,
                'mean_neg': mean_neg,
                'diff': mean_pos - mean_neg
            }

    # 按正负差异降序排列，取 Top-k
    sorted_heads = sorted(head_stats.keys(), key=lambda k: head_stats[k]['diff'], reverse=True)
    selected_heads = sorted_heads[:top_k]
    return selected_heads, head_stats
```

### 3.5 投票式幻觉检测

```python
def voting_prediction(probes, features_dict, selected_heads, threshold=0.5):
    """
    5 个单头分类器投票决策。

    Args:
        probes: dict {(layer, head): trained_keras_model}
        features_dict: dict {(layer, head): np.array [N, d_model]}
        selected_heads: list of (layer, head)
        threshold: 单个分类器的判定阈值
    Returns:
        predictions: np.array [N] — 最终预测（1=正确，0=幻觉）
        individual_preds: dict {(layer, head): np.array [N]} — 各分类器的预测
    """
    individual_preds = {}
    for head_key in selected_heads:
        prob = probes[head_key].predict(features_dict[head_key], verbose=0).flatten()
        individual_preds[head_key] = (prob > threshold).astype(int)

    # 多数投票
    pred_matrix = np.stack(list(individual_preds.values()), axis=0)  # [5, N]
    vote_count = pred_matrix.sum(axis=0)  # [N]
    predictions = (vote_count >= 3).astype(int)  # 超过半数判定为正确

    return predictions, individual_preds
```

### 3.6 TriviaQA 数据处理与生成

```python
from datasets import load_dataset

# 加载 TriviaQA
triviaqa = load_dataset("trivia_qa", "rc.nocontext")
train_data = triviaqa["train"]
val_data = triviaqa["validation"]

# 数据格式示例：
# {
#   "input": [
#     {"role": "system", "content": "Follow the given examples and answer the question."},
#     {"role": "user", "content": "Who was the man behind The Chipmunks?"}
#   ],
#   "ideal": ["David Seville", "david seville"]
# }

# 划分三个集合
# 集合 A（头筛选）：从 train 中采样 300 条
set_a = train_data.shuffle(seed=42).select(range(300))
# 集合 B（探针训练）：从 train 中采样后续 500 条（不与 A 重叠）
set_b = train_data.shuffle(seed=42).select(range(300, 800))
# 集合 C（验证）：从 validation 中采样 200 条
set_c = val_data.shuffle(seed=42).select(range(200))

# 使用数据集自带的 input 字段作为 Prompt（chat 格式）
# 直接传入 tokenizer.apply_chat_template 进行格式化

# 自动标注：检查生成答案是否包含在 ideal 列表中
def is_correct(generated_answer, ideal_answers):
    gen = generated_answer.strip().lower()
    return any(gt.lower() in gen or gen in gt.lower() for gt in ideal_answers)
```

## 4. 预期产出

### 4.1 代码文件

| 文件名                         | 功能                                                                  |
| ------------------------------ | --------------------------------------------------------------------- |
| `Qwen2_TriviaQA_generate.py` | TriviaQA 数据下载、划分、答案生成与自动标注                           |
| `Qwen2_LogitAttribution.py`  | Hook 注册 → 影响向量提取 → Logit 贡献计算 → 基于正负样本模式筛选头 |
| `Qwen2_HeadProbe.py`         | 基于 Top-k 头的探针训练、投票式幻觉检测评估                           |

### 4.2 数据与结果文件

| 文件名                                           | 内容                                      |
| ------------------------------------------------ | ----------------------------------------- |
| `datasets/triviaqa_set_a.csv`                  | 头筛选用数据（带生成答案与正确/幻觉标签） |
| `datasets/triviaqa_set_b.csv`                  | 探针训练用数据                            |
| `datasets/triviaqa_set_c.csv`                  | 端到端验证用数据                          |
| `processed_datasets/head_contributions.csv`    | 448 个头的正/负样本平均贡献量及差异排名   |
| `processed_datasets/top5_heads.json`           | Top-5 头的 (layer, head) 索引及筛选依据   |
| `processed_datasets/head_probe_metrics.csv`    | 每个头的探针评估指标 + 投票集成指标       |
| `processed_datasets/fig_head_heatmap.png`      | 正负样本贡献差异热力图（层×头）          |
| `processed_datasets/fig_voting_comparison.png` | 单头 vs 投票性能对比图                    |

### 4.3 预期对比

| 方法                       | 特征来源                 | 预期表现                                |
| -------------------------- | ------------------------ | --------------------------------------- |
| PPL                        | 生成概率                 | AUC ~0.70-0.78（基线）                  |
| SAPLMA (L18)               | 全层隐藏状态             | AUC ~0.83（基线）                       |
| 单头探针 (Top-1)           | 单个注意力头影响向量     | 预期接近 SAPLMA                         |
| 单头探针 (Top-5 平均)      | 5 个头各自独立检测的平均 | 预期与 SAPLMA 可比                      |
| **投票集成 (Top-5)** | **5 个头投票**     | **预期优于单头，可能超越 SAPLMA** |

## 5. 风险与备选方案

| 风险                                                                  | 应对策略                                                                               |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 满足"正样本正贡献、负样本负贡献"条件的头不足 5 个                     | 放宽条件为"正负样本贡献方向相反"（允许正样本负贡献、负样本正贡献），按绝对差异排序选取 |
| Qwen2 注意力模块内部结构与预期不符，hook 难以提取 `o_proj` 前的输出 | 改用自定义 wrapper 替换 `forward` 方法，手动分离头输出                               |
| TriviaQA 自动标注噪声大                                               | 增加人工抽检比例；改用更宽松的匹配策略（如 Jaccard 相似度 > 0.5）                      |
| 单头探针性能远低于 SAPLMA                                             | 增加 k 值（如 10）或尝试多头联合特征作为补充方案                                       |
| 投票结果过于一致（所有头输出相同）                                    | 分析各头预测的相关性，若高度相关则说明筛选标准需调整，应选出互补性强的头               |
| QA 生成时模型不产生有效答案（如输出空或重复）                         | 在生成后过滤无效样本，确保每个集合的有效样本量充足                                     |
