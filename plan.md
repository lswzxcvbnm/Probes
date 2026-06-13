# 基于逐头分类器筛选与投票集成的幻觉检测研究计划

## 1. 研究背景与动机

SAPLMA 方法通过提取隐藏层表征训练探针来检测幻觉，已在阶段二中验证了其有效性（Layer 18 平均 AUC 0.834）。然而，SAPLMA 将整个隐藏状态（1536 维）作为黑盒特征，并未揭示模型内部哪些组件对"知道答案是否正确"起到了关键作用，也无法解释幻觉检测的可归因性。

本阶段的目标是**从注意力头层面实现可解释的幻觉检测**：在 QA 生成任务上，为每个注意力头训练独立的二分类器，以该头的影响向量（influence vector）作为特征来判断生成答案是否正确。通过比较 336 个头分类器的判别性能（验证集 AUC），筛选出最具判别力的 Top-k 个注意力头，最终通过投票集成进行幻觉检测。

### 核心思想

在 Transformer 的前向传播中，每个注意力头 $h$ 在残差流中添加一个增量向量：

$$
\Delta_h = O_h W_O^h
$$

其中 $O_h$ 为头 $h$ 的注意力输出，$W_O^h$ 为输出投影矩阵中对应头 $h$ 的切片。该向量 $\Delta_h \in \mathbb{R}^{d_{model}}$ 即为头 $h$ 的**影响向量**，直接参与构成最终的隐藏状态，进而通过 unembedding 矩阵影响下一个 token 的 logit 分布。

**关键假设**：如果模型在生成答案时"知道"答案是否正确，那么这种"知识"应体现在特定注意力头的影响向量中——正确生成与幻觉生成时，这些头的影响向量应呈现可区分的模式。

**方法**：为全部 336 个注意力头（28 层 × 12 头）各训练一个二分类器，以影响向量为特征，通过验证集 AUC 筛选 Top-k 头。筛选阶段的 336 个分类器仅用于头选择；评估阶段在选定头上重新训练探针，通过多数投票得到最终检测结果。与 SAPLMA 使用完整隐藏状态（1536 维）不同，本方法仅使用 5/336 个头的影响向量即可实现更优的检测性能，同时揭示了对幻觉检测最关键的注意力头位置。

## 2. 技术方案

### 2.1 模型与数据

| 项目             | 选择       | 说明                                                     |
| ---------------- | ---------- | -------------------------------------------------------- |
| 基座模型         | Qwen2-1.5B | 与阶段一、二一致，28 层 Transformer，每层 12 个注意力头  |
| 训练集（集合 A） | TriviaQA   | ~500 条，用于训练 336 个单头二分类器                     |
| 验证集（集合 B） | TriviaQA   | ~300 条，用于评估分类器 AUC 并筛选 Top-k 头             |
| 测试集（集合 C） | TriviaQA   | ~200 条，用于最终端到端幻觉检测性能评估                  |

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

### 2.2 注意力头特征提取与分类器筛选（核心步骤）

#### 步骤 1：生成答案并划分数据集

1. 对 TriviaQA 数据集中的每个问题，使用 Qwen2-1.5B 生成答案（greedy decoding）
2. 将生成答案与标准答案（`ideal` 字段，为可接受答案别名列表）进行模糊匹配，划分为：
   - **正样本（正确生成）**：生成答案包含在 `ideal` 列表中
   - **负样本（幻觉生成）**：生成答案不在 `ideal` 列表中
3. 将全部有效样本随机划分为三个互不重叠的子集：训练集 A（~500 条）、验证集 B（~300 条）、测试集 C（~200 条）

#### 步骤 2：提取全部 336 个头的影响向量

在 Qwen2-1.5B 的每一层注意力模块上注册 forward hook，提取每个头在 `o_proj` 前的输出，计算影响向量。

具体而言，对于 Qwen2 的注意力模块 `model.model.layers[l].self_attn`：

- 输入：隐藏状态 $x \in \mathbb{R}^{B \times T \times d_{model}}$
- 每个头 $h$ 计算：
  - $Q_h = x W_Q^h$, $K_h = x W_K^h$, $V_h = x W_V^h$
  - 注意力权重 $A_h = \text{softmax}(Q_h K_h^T / \sqrt{d_k})$
  - 头输出 $O_h = A_h V_h$
  - **影响向量** $\Delta_h = O_h W_O^h$，其中 $W_O^h$ 是 $W_O$ 对应第 $h$ 个头的切片

> **实现方案**：hook `o_proj` 的输入（形状 `[B, T, num_heads * head_dim]`），按 `head_dim` 切分得到每个头的输出 $O_h$，再与 `o_proj` 权重矩阵的对应切片相乘得到 $\Delta_h$。

对于每个输入样本，**取生成答案的最后一个 token 位置**的影响向量 $\Delta_h[\text{ans\_last\_pos}] \in \mathbb{R}^{d_{model}}$ 作为该头的特征。答案末尾 token 通常承载了最多的语义信息，是模型"决定说什么"的关键位置。

共提取 $28 \times 12 = 336$ 个头的 1536 维特征向量，用于后续分类器训练。

#### 步骤 3：为每个注意力头训练二分类器，筛选 Top-k 头

**核心思路**：直接为每个注意力头训练一个独立的二分类器，以该头的影响向量作为特征，通过分类器在验证集上的性能来筛选最具判别力的头。

**训练流程**：

1. **特征提取**：对训练集（集合 A）中的每个样本，取每个注意力头 $h$ 在答案最后一个 token 位置的影响向量 $\Delta_h[\text{ans\_last\_pos}] \in \mathbb{R}^{d_{model}}$ 作为特征
2. **训练分类器**：为每个头 $h$（共 336 个）独立训练一个二分类器（三层前馈网络）
3. **验证评估**：在验证集上评估每个分类器的 **AUC** 作为头的判别力指标
4. **筛选 Top-k**：按验证集 AUC 降序排列，选取 **Top-k 个头**（$k=5$）

$$
\text{score}(h) = \text{AUC}_h^{\text{val}}
$$

$$
\text{Top-k} = \arg\text{top-k}_{h} \text{score}(h)
$$

> **为什么使用分类器性能而非贡献方向？**
> - 分类器性能直接衡量头的判别力，比基于统计假设的方向性筛选更可靠
> - 不依赖"正确样本贡献为正、幻觉样本贡献为负"的强假设，适用性更广
> - 分类器可以学习更复杂的非线性决策边界，捕捉方向性筛选可能遗漏的模式
> - AUC 指标对类别不平衡具有鲁棒性

### 2.3 基于 Top-k 头的投票式幻觉检测

步骤 3 中训练的 336 个分类器仅用于**筛选 Top-k 头**（按验证集 AUC 排序）。筛选完成后，在 `Qwen2_HeadProbe.py` 中为 Top-k 头**重新训练**探针用于最终评估和投票。这样做的原因是：筛选阶段的分类器使用 50 个 epoch 和 early stopping，而评估阶段的探针使用更少的 epoch（5 次）配合多次随机重启（3 次）和最优阈值选择，以获得更稳健的评估结果。

#### 单头探针重新训练

在 `Qwen2_HeadProbe.py` 中，对 Top-k 头重新训练探针：

1. **特征**：该头在答案最后一个 token 位置的影响向量 $\Delta_{h_j}[\text{ans\_last\_pos}] \in \mathbb{R}^{d_{model}}$（维度 = 1536）
2. **标签**：生成答案的正确性标签（1=正确生成，0=幻觉生成）
3. **探针架构**：与步骤 3 相同的三层前馈网络
   - `Dense(256, relu) → Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)`
   - 优化器：Adam，损失函数：binary cross-entropy
4. **训练策略**：StandardScaler 归一化 + 3 次随机重启（每次 5 个 epoch），选择训练集上准确率最高的模型
5. **阈值选择**：通过 ROC 曲线在训练集上寻优阈值，使准确率最大化

#### 投票机制

对于每个测试样本，5 个独立探针各自输出一个预测概率 $p_j \in [0, 1]$：

$$
\text{vote}(\mathbf{p}) = \begin{cases} 1 \text{ (正确)} & \text{if } \sum_{j=1}^{5} \mathbb{1}[p_j > \tau_j] \geq 3 \\ 0 \text{ (幻觉)} & \text{otherwise} \end{cases}
$$

其中 $\tau_j$ 为第 $j$ 个探针在训练集上通过 ROC 曲线寻优的阈值（实际实现中使用各探针最优阈值的平均值 $\bar{\tau}$ 作为统一阈值）。即多数投票：超过半数的探针判定为正确，则最终判定为正确；否则判定为幻觉。

> **投票的优势**：
>
> - 降低单个头的噪声影响，提高鲁棒性
> - 每个头关注不同层面的信息（如事实记忆、推理路径、置信度），投票融合多角度判断
> - 可分析单个头的独立性能与投票后的集成性能差异

### 2.4 数据划分与端到端评估

#### 数据划分

将 TriviaQA 数据划分为三个互不重叠的子集：

| 子集   | 用途     | 样本量  | 说明                                                          |
| ------ | -------- | ------- | ------------------------------------------------------------- |
| 集合 A | 训练集   | ~500 条 | 用于训练 336 个单头二分类器                                   |
| 集合 B | 验证集   | ~300 条 | 用于评估每个头的分类器 AUC，筛选 Top-k 头                     |
| 集合 C | 测试集   | ~200 条 | 用于最终评估投票式幻觉检测的端到端性能                        |

> 所有子集均通过 QA 生成 → 自动标注流程，得到每条样本的正确/幻觉标签。
> **注意**：集合 A 和 B 的用途与旧方案不同——旧方案中集合 A 用于计算贡献量筛选头，集合 B 用于训练探针；新方案中集合 A 用于训练全部 336 个分类器，集合 B 用于评估并筛选 Top-k 头。

#### 端到端评估流程

对测试集（集合 C）中的每个问答对：

1. 用 Qwen2-1.5B 生成答案（与训练时一致的 Prompt 和解码策略）
2. 自动标注答案正确性作为真实标签
3. 提取 Top-k 头在答案最后一个 token 位置的影响向量
4. 用重新训练的 k 个单头探针分别预测（每个探针输出概率，经最优阈值二值化）
5. 通过多数投票得到最终预测
6. 计算 **Accuracy、Precision、Recall、F1、AUC**

#### 对比基线

- **PPL 方法**：在同一测试集上，用负对数似然作为幻觉分数，ROC 寻优阈值
- **SAPLMA 方法**：在同一测试集上，提取 Layer 18 隐藏状态（完整残差流，维度 1536）训练探针，作为性能上限参考
- **单头探针（无投票）**：每个头的独立性能，分析投票的增益
- **求和探针（Top-5 求和）**：5 个头的影响向量逐元素求和，训练单个探针
- **投票集成（Top-5）**：5 个单头探针多数投票

> **对比分析**：
> - **Single vs Summed**：单头探针各自独立，求和探针将 5 个头的信号合并。若 Summed > Single，说明多头信号互补
> - **Summed vs Voting**：两者都融合了 5 个头的信息，但方式不同——求和是特征级融合，投票是决策级融合
> - **Summed vs SAPLMA**：求和向量仅包含 5 个头的贡献（部分残差流），SAPLMA 包含全部头 + MLP + 残差。差距反映了未选中头和 MLP 的信息量

## 3. 关键实现细节

### 3.1 Qwen2 注意力模块结构

Qwen2-1.5B 的注意力实现位于 `modeling_qwen2.py` 中的 `Qwen2Attention` 类：

```python
class Qwen2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        self.num_heads = config.num_attention_heads  # 12
        self.head_dim = config.hidden_size // config.num_attention_heads  # 128
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
```

**提取影响向量的关键**：在 `o_proj` 之前，`attn_output` 的形状为 `[B, T, num_heads * head_dim]`，可以按 `head_dim=128` 切分得到每个头的输出，再分别与 `o_proj.weight` 的对应列相乘。

### 3.2 Hook 实现策略

采用**单阶段 Hook** 方案：逐样本注册全部 28 层的 hook，提取 `o_proj` 输入，然后在 CPU 上计算所有 336 个头的影响向量。

```python
# 每样本注册 hook → 提取 o_proj 输入 → 计算全部头影响向量 → 释放
for idx in range(N):
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

    model(input_ids)  # forward pass

    # 计算全部 336 个头在答案位置的影响向量
    for layer in range(num_layers):
        attn_out = o_proj_inputs[layer]
        W_O = model.model.layers[layer].self_attn.o_proj.weight.detach().float().cpu().numpy()
        for head in range(num_heads):
            head_out = attn_out[answer_last_pos, head*head_dim:(head+1)*head_dim]
            W_O_h = W_O[:, head*head_dim:(head+1)*head_dim]
            delta_h = head_out @ W_O_h.T  # [d_model]
            all_features[(layer, head)].append(delta_h)

    for h in hook_handles: h.remove()
    del o_proj_inputs  # 释放内存
```

> **内存分析**：每样本存储 336 × 1536 × 4B ≈ 2MB（仅答案位置），500 样本总计 ~1GB。hook 存储为中间量（28 层 × T × 1536 × 4B），释放后即可复用。

### 3.3 影响向量计算

对全部 336 个头计算答案位置的影响向量，作为分类器的输入特征：

```python
def compute_all_head_features(o_proj_inputs, answer_last_pos, model):
    """
    计算全部 336 个头在答案位置的影响向量。

    Args:
        o_proj_inputs: dict {layer_idx: np.array [T, num_heads * head_dim]}
        answer_last_pos: int — 答案最后一个 token 的位置
        model: the model (for o_proj weights)
    Returns:
        features: dict {(layer, head): np.array [d_model]}
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    features = {}

    for layer_idx, attn_out in o_proj_inputs.items():
        W_O = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().float().cpu().numpy()
        for head in range(num_heads):
            head_out = attn_out[answer_last_pos, head*head_dim:(head+1)*head_dim]
            W_O_h = W_O[:, head*head_dim:(head+1)*head_dim]
            features[(layer_idx, head)] = head_out @ W_O_h.T  # [d_model]

    return features
```

### 3.4 基于分类器性能的注意力头筛选

实际实现将训练和筛选分为两个独立函数（`Qwen2_HeadsAttribution.py`）：

```python
def build_head_probe(input_dim=1536):
    """构建三层前馈网络分类器。"""
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
    为每个注意力头训练二分类器。

    Args:
        features_train: dict {(layer, head): np.array [N_train, d_model]}
        labels_train: np.array [N_train]
        num_layers: 层数
        num_heads: 每层头数
        epochs: 最大训练 epoch 数（配合 early stopping）
        batch_size: 训练 batch size
    Returns:
        all_classifiers: dict {(layer, head): (model, scaler)}
    """
    all_classifiers = {}

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

    return all_classifiers


def select_heads_by_auc(all_classifiers, features_val, labels_val, top_k=5):
    """
    在验证集上评估每个头的分类器 AUC，筛选 Top-k 头。

    Args:
        all_classifiers: dict {(layer, head): (model, scaler)}
        features_val: dict {(layer, head): np.array [N_val, d_model]}
        labels_val: np.array [N_val]
        top_k: 选取的头数量
    Returns:
        selected_heads: list of (layer, head) — 按 AUC 降序排列的 Top-k 头
        head_aucs: dict {(layer, head): float} — 每个头的验证集 AUC
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
```

> **筛选阶段与评估阶段的区别**：上述 336 个分类器仅用于筛选 Top-k 头。筛选完成后，在 `Qwen2_HeadProbe.py` 中为 Top-k 头重新训练探针用于最终评估，训练策略不同（5 epoch × 3 次随机重启 + 最优阈值选择）。

### 3.5 投票式幻觉检测

投票使用 `Qwen2_HeadProbe.py` 中重新训练的探针（非 HeadsAttribution 中的 336 个分类器）：

```python
def voting_prediction(probes, features_dict, selected_heads, threshold=0.5):
    """
    k 个单头探针投票决策。

    Args:
        probes: dict {head_key: trained_keras_model} — 重新训练的探针
        features_dict: dict {head_key: np.array [N, d_model]} — 已归一化的特征
        selected_heads: list of (layer, head)
        threshold: 单个探针的判定阈值（通过 ROC 曲线在训练集上寻优）
    Returns:
        predictions: np.array [N] — 最终预测（1=正确，0=幻觉）
        individual_preds: dict {head_key: np.array [N]} — 各探针的预测
    """
    individual_preds = {}
    for layer, head in selected_heads:
        key = f"L{layer}_H{head}"
        if key in probes and key in features_dict:
            prob = probes[key].predict(features_dict[key], verbose=0).flatten()
            individual_preds[key] = (prob > threshold).astype(int)

    # 多数投票
    pred_matrix = np.stack(list(individual_preds.values()), axis=0)  # [k, N]
    vote_count = pred_matrix.sum(axis=0)  # [N]
    k = len(selected_heads)
    majority = (k + 1) // 2  # e.g., 3 for k=5
    predictions = (vote_count >= majority).astype(int)

    return predictions, individual_preds
```

> **注意**：实际实现中，投票使用的阈值是各探针在训练集上通过 ROC 曲线寻优的阈值的平均值，而非固定的 0.5。

### 3.6 TriviaQA 数据处理与生成

```python
# 加载本地 TriviaQA jsonl 文件
all_samples = load_triviaqa_data("datasets/triviaqa/trivia_qa/test.jsonl")

# 可选：子采样以控制生成时间（默认 2000 条，足够 500+300+200）
if max_samples and max_samples < len(all_samples):
    indices = rng.choice(len(all_samples), size=max_samples, replace=False)
    all_samples = [all_samples[i] for i in indices]

# 为全部样本生成答案并自动标注
all_results = generate_and_label(model, tokenizer, all_samples)

# 划分三个集合（无需刻意平衡，分类器可处理类别不平衡）
# Set A: 训练集（~500 条），用于训练 336 个单头二分类器
# Set B: 验证集（~300 条），用于评估分类器 AUC 并筛选 Top-k 头
# Set C: 测试集（~200 条），用于端到端评估投票式幻觉检测
set_a = all_results[:500]
set_b = all_results[500:800]
set_c = all_results[800:1000]

# 自动标注：检查生成答案是否包含在 ideal 列表中
def is_correct(generated_answer, ideal_answers):
    gen = generated_answer.strip().lower()
    return any(gt.lower() in gen or gen in gt.lower() for gt in ideal_answers)
```

## 4. 预期产出

### 4.1 代码文件

| 文件名                         | 功能                                                                          |
| ------------------------------ | ----------------------------------------------------------------------------- |
| `Qwen2_TriviaQA_generate.py` | TriviaQA 数据加载、子采样、答案生成、自动标注、随机划分三个集合             |
| `Qwen2_HeadsAttribution.py`  | 提取全部 336 头特征 → 训练分类器 → 按 AUC 筛选 Top-k 头 → 保存分类器     |
| `Qwen2_HeadProbe.py`         | 基于 Top-k 头的探针训练（含 StandardScaler 归一化）、投票式幻觉检测评估     |

### 4.2 数据与结果文件

| 文件名                                           | 内容                                      |
| ------------------------------------------------ | ----------------------------------------- |
| `processed_datasets/triviaqa_set_a.csv`        | 训练集（~500 条，用于训练 336 个分类器）  |
| `processed_datasets/triviaqa_set_b.csv`        | 验证集（~300 条，用于评估 AUC 筛选头）    |
| `processed_datasets/triviaqa_set_c.csv`        | 测试集（~200 条，端到端评估）             |
| `processed_datasets/head_aucs.csv`             | 336 个头的分类器验证集 AUC 排名           |
| `processed_datasets/top5_heads.json`           | Top-5 头的 (layer, head) 索引及 AUC       |
| `processed_datasets/head_classifiers/`         | 336 个分类器中 Top-5 头的模型 (.h5) 和 scaler（用于筛选阶段） |
| `processed_datasets/head_probes/`              | Top-5 头重新训练的探针模型 (.h5)（用于评估和投票） |
| `processed_datasets/set_a_all_features.npz`    | 集合 A 中 Top-5 头的影响向量特征          |
| `processed_datasets/set_b_selected_features.npz` | 集合 B 中 Top-5 头的影响向量特征        |
| `processed_datasets/head_probe_metrics.csv`    | 每个头的探针评估指标 + 投票集成指标       |
| `processed_datasets/fig_head_auc_heatmap.png`  | 336 个头的分类器 AUC 热力图（层×头）      |
| `processed_datasets/fig_voting_comparison.png` | 单头 vs 投票性能对比图                    |

### 4.3 实验结果

**筛选出的 Top-5 注意力头**（按验证集 AUC 排序）：

| 排名 | 注意力头 | 验证集 AUC |
| ---- | -------- | ---------- |
| 1    | L15_H6   | 0.7823     |
| 2    | L13_H11  | 0.7793     |
| 3    | L15_H9   | 0.7781     |
| 4    | L16_H8   | 0.7727     |
| 5    | L13_H6   | 0.7606     |

> Top-5 头均位于**第 13-16 层**（模型中上层区域），与 SAPLMA 最优层（Layer 18）相近。

**测试集（集合 C，203 条）端到端评估结果**：

| 方法              | Accuracy | Precision | Recall | F1    | AUC   |
| ----------------- | -------- | --------- | ------ | ----- | ----- |
| PPL               | 0.625    | 0.889     | 0.098  | 0.176 | 0.539 |
| SAPLMA-L18        | 0.680    | 0.667     | 0.439  | 0.529 | 0.723 |
| 单头 L15_H6       | 0.675    | 0.579     | 0.756  | 0.656 | 0.755 |
| 单头 L13_H11      | 0.660    | 0.613     | 0.463  | 0.528 | 0.708 |
| 单头 L15_H9       | 0.680    | 0.598     | 0.671  | 0.632 | 0.741 |
| 单头 L16_H8       | 0.645    | 0.590     | 0.439  | 0.503 | 0.704 |
| 单头 L13_H6       | 0.685    | 0.607     | 0.659  | 0.632 | 0.733 |
| **求和探针 (Top-5)** | **0.710** | **0.676** | **0.561** | **0.613** | **0.776** |
| **投票集成 (Top-5)** | 0.685    | 0.620     | 0.598  | 0.609 | 0.762 |

> **注**：投票集成使用 `Qwen2_HeadProbe.py` 中重新训练的探针（5 epoch × 3 次随机重启 + 最优阈值），而非 `Qwen2_HeadsAttribution.py` 中用于筛选头的 336 个分类器。

**关键发现**：

- **求和探针以 AUC 0.776 和 Accuracy 0.710 取得最优表现**，仅使用 5/336 个头的影响向量即超越 SAPLMA（AUC 0.723）
- 投票集成（AUC 0.762）接近求和探针但略低，说明特征级融合优于决策级融合
- PPL 召回率极低（0.098），几乎无法检测幻觉
- 单头探针中 L15_H6 召回率最高（0.756），L13_H6 综合最均衡

## 5. 风险与应对

| 风险                                                    | 应对策略                                                                                 | 实验验证结果                                                                                      |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 选出的头区分力不足                                      | 增加 k 值或尝试多头联合；改用分类器 AUC 筛选替代方向性筛选                               | **已解决**：分类器 AUC 筛选出的 Top-5 头（AUC 0.76-0.78）表现显著优于旧的方向性筛选               |
| hook 难以提取 `o_proj` 前的输出                        | hook `o_proj` 的输入即可获取拼接后的头输出                                               | **已解决**：hook `o_proj` 输入稳定可行                                                            |
| TriviaQA 自动标注噪声大                                | 增加人工抽检比例；改用更宽松的匹配策略                                                   | 部分验证：PPL 召回率极低（0.098），说明标注噪声可能影响基线，但探针方法鲁棒                       |
| 单头探针性能远低于 SAPLMA                              | 增加 k 值或尝试多头联合特征                                                              | **已解决**：Top-1 单头（L15_H6, AUC 0.755）已接近 SAPLMA（0.723），求和探针（0.776）超越 SAPLMA  |
| 投票结果过于一致（所有头输出相同）                      | 分析各头预测的相关性，选出互补性强的头                                                   | 部分验证：投票（0.762）略低于求和（0.776），说明头间存在冗余，但仍有效                            |
| QA 生成时模型不产生有效答案                             | 在生成后过滤无效样本                                                                     | **已解决**：生成流程稳定，502/304/203 条有效样本                                                  |
