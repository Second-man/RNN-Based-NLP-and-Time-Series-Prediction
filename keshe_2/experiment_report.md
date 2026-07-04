# 基于 Keras 的文本类项目实验报告

## 一、实验目的

1. 掌握 **SimpleRNN** 在文本分类任务中的基本使用方法；
2. 掌握 **LSTM** 在时间序列预测（股票价格）任务中的建模流程；
3. 掌握基础 **Seq2Seq** 架构在机器翻译（英-法 / 法-英）中的编解码实现；
4. 掌握加入 **Attention 机制** 的 Encoder-Decoder 模型在机器翻译中的改进效果；
5. 对比不同模型架构在相同数据集上的表现差异，分析各模型的适用场景与优劣。

## 二、实验环境

| 项目 | 内容 |
|------|------|
| 操作系统 | Ubuntu (Linux) |
| 深度学习框架 | TensorFlow 2.21.0 |
| 硬件加速 | GPU (NVIDIA, CUDA) |
| Python 版本 | Python 3.x |
| 核心依赖 | tensorflow, pandas, pyarrow, scikit-learn, matplotlib |

## 三、实验方案设计

### 数据集说明

| 任务 | 数据集 | 来源 | 规模 |
|------|--------|------|------|
| 任务1: RNN 文本分类 | 英法平行语料中的英文句子 | `task4/.../train-*.parquet` | 1000 句采样 |
| 任务2: LSTM 股票预测 | 股票历史价格序列 | `stock_data.csv` (TSLA) | 1858 行 |
| 任务3: 基础 Seq2Seq 翻译 | 英法平行句对 | `fra.txt`（约 24 万句） | 150000 对 |
| 任务4: Attention 编解码翻译 | 英法平行句对 | `fra.txt`（约 24 万句） | 150000 对 |

### 模型架构设计

#### 任务1 — SimpleRNN 文本分类

```text
Input (string) → TextVectorization → Embedding(64) → SimpleRNN(64) → Dropout(0.2) → Dense(sigmoid)
```

任务：判断英文句子是否为长句（词数 > 中位数）。二分类。

#### 任务2 — LSTM 股票价格预测

```text
Input (window=60) → LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dense(1)
```

任务：基于过去 60 个交易日的收盘价预测下一个交易日收盘价。回归任务。

#### 任务3 — 基础 Seq2Seq 机器翻译

```text
Encoder: source tokens → Embedding(192) → LSTM(192) → [h, c] states
Decoder: target tokens → Embedding(192) → LSTM(192, init_state=[h, c]) → Dense(Softmax)
```

任务：英法互译。训练时使用 Teacher Forcing，推理时逐词生成。

#### 任务4 — Attention Encoder-Decoder 机器翻译

```text
Encoder: source → Embedding(192) → Bidirectional LSTM(192) → [h, c] states + encoder_outputs
Decoder: target → Embedding(192) → LSTM(384) → Attention(encoder_outputs) → Concat → Dense(Softmax)
```

任务：英法互译。相比任务3增加了双向 LSTM 编码器和 Attention 注意力层。推理支持 Greedy 和 Beam Search 两种解码策略。

## 四、实验过程

### 4.1 任务1 — RNN 文本分类

**运行命令**：

```bash
python task1_rnn/01_rnn_text_sequence.py --max_samples 1000 --epochs 10
```

**模型结构**：

| 层 (Layer) | 输出维度 | 参数 |
|-----------|---------|------|
| TextVectorization | (30,) | 0 |
| Embedding | (30, 64) | 320,000 |
| SimpleRNN | 64 | 8,256 |
| Dropout | 64 | 0 |
| Dense (sigmoid) | 1 | 65 |

**训练曲线**：

![RNN Accuracy曲线](asset/task1_rnn_accuracy_curve.png)

![RNN Loss曲线](asset/task1_rnn_loss_curve.png)

**预测示例**：

```
text: M. Thomsen placed his services at our disposal...
word_count: 24
predicted_probability_long: 0.9898
predicted_label: 1 (长句)

text: "And you saw him frequently?"
word_count: 5
predicted_probability_long: 0.0032
predicted_label: 0 (短句)
```

### 4.2 任务2 — LSTM 股票价格预测

**运行命令**：

```bash
python task2_lstm/02_lstm_stock_prediction.py --stock TSLA --window_size 60 --epochs 30
```

**模型结构**：

| 层 (Layer) | 输出维度 | 参数 |
|-----------|---------|------|
| LSTM (64, return_sequences) | (60, 64) | 16,896 |
| Dropout (0.2) | (60, 64) | 0 |
| LSTM (32) | 32 | 12,416 |
| Dense | 1 | 33 |

**训练 Loss 曲线**：

![LSTM Loss曲线](asset/task2_lstm_loss_curve.png)

**预测对比曲线**（TSLA 收盘价真实值 vs 预测值）：

![LSTM 预测曲线](asset/task2_lstm_prediction_curve.png)

**预测示例**（部分日期）：

| Date | Actual Close | Predicted Close |
|------|-------------|-----------------|
| 2016-06-10 | 218.79 | 230.74 |
| 2016-12-30 | 213.69 | 219.12 |
| 2017-03-31 | 278.30 | 270.10 |
| 2017-06-14 | 380.66 | 350.05 |
| 2017-09-15 | 379.81 | 346.34 |
| 2017-11-10 | 302.99 | 298.56 |

### 4.3 任务3 — 基础 Seq2Seq 机器翻译

**运行命令**：

```bash
# en → fr
python task3_machine_translation/03_seq2seq_translation.py \
    --direction en-fr --max_samples 150000 --epochs 25

# fr → en
python task3_machine_translation/03_seq2seq_translation.py \
    --direction fr-en --max_samples 150000 --epochs 25
```

**模型结构**（en-fr 方向）：

```
Encoder:
  Embedding(8000→192) → LSTM(192) → [h, c]

Decoder:
  Embedding(8000→192) → LSTM(192, init_state=[h,c]) → Dense(8000, Softmax)
```

**训练曲线（en→fr）**：

![Seq2Seq en-fr Accuracy](asset/task3_en_fr_accuracy_curve.png)

![Seq2Seq en-fr Loss](asset/task3_en_fr_loss_curve.png)

**训练曲线（fr→en）**：

![Seq2Seq fr-en Accuracy](asset/task3_fr_en_accuracy_curve.png)

![Seq2Seq fr-en Loss](asset/task3_fr_en_loss_curve.png)

**翻译示例（en→fr）**：

| Source | Target | Predicted |
|--------|--------|-----------|
| i bumped into tom in the supermarket yesterday | je suis tombé sur tom au supermarché | je suis tombé sur le bureau hier |
| would you like to dance | voudrais tu danser | aimerais tu danser |
| stop overreacting | cessez d'exagérer | cesse de réagir de façon excessive |
| are your parents sleeping | est ce que vos parents dorment | tes parents dorment |
| he changed schools last year | il a changé d'école l'année dernière | il a changé de l'année dernière |

**翻译示例（fr→en）**：

| Source | Target | Predicted |
|--------|--------|-----------|
| soyez davantage flexibles | be more flexible | be more precise |
| pourquoi êtes vous toutes là | why are you all here | why are you all here |
| tom a dit que la soupe était trop chaude | tom said that the soup was too hot | tom said that the soup was too hot |
| tom s'est rendu | tom has surrendered | tom surrendered |
| écris une ligne sur deux | write on every other line | write a line in two |

### 4.4 任务4 — Attention Encoder-Decoder 机器翻译

**运行命令**：

```bash
# en → fr
python task4_encoder_decoder_translation/04_attention_encoder_decoder_translation.py \
    --direction en-fr --max_samples 150000 --epochs 25

# fr → en
python task4_encoder_decoder_translation/04_attention_encoder_decoder_translation.py \
    --direction fr-en --max_samples 150000 --epochs 25
```

**模型结构**：

```
Encoder:
  Embedding(8000→192) → Bidirectional LSTM(192) → concat states + encoder_outputs

Decoder:
  Embedding(8000→192) → LSTM(384, init_state=[h,c]) → Attention(encoder_outputs)
  → Concat → Dropout(0.2) → Dense(8000, Softmax)
```

**训练曲线（en→fr）**：

![Attention en-fr Accuracy](asset/task4_en_fr_accuracy_curve.png)

![Attention en-fr Loss](asset/task4_en_fr_loss_curve.png)

**训练曲线（fr→en）**：

![Attention fr-en Accuracy](asset/task4_fr_en_accuracy_curve.png)

![Attention fr-en Loss](asset/task4_fr_en_loss_curve.png)

**翻译示例（en→fr，含 Greedy 和 Beam Search 对比）**：

| Source | Target | Greedy | Beam (k=5) |
|--------|--------|--------|------------|
| what part of australia do you come from | de quel coin de l'australie viens tu | à quelle partie de l'australie viens tu | à quelle partie de l'australie viens tu |
| may i please have your telephone number | pourrais je avoir votre numéro de téléphone s'il vous plaît | puis je avoir votre numéro de téléphone | puis je avoir votre numéro de téléphone |
| the political situation has changed | la situation politique a changé | la situation depuis | la situation depuis |
| tom said he'd play tennis with us | tom a dit qu'il jouerait au tennis avec nous | tom a dit qu'il joue au tennis avec nous | tom a dit qu'il joue au tennis avec nous |

**翻译示例（fr→en，含 Greedy 和 Beam Search 对比）**：

| Source | Target | Greedy | Beam (k=5) |
|--------|--------|--------|------------|
| nos invitées arrivent | our guests are arriving | our invited will come | our invited will come |
| est ce que tu sens une odeur de gaz | do you smell gas | do you feel a gas smell | do you feel a gas smell |
| merci beaucoup d'avance | thank you in advance | thank you very much in advance | thank you very much in advance |
| j'ai trois fois plus de livres que tom | i have three times as many books as tom | i have three times as many books as tom | i have three times as many books as tom |
| je veux simplement dire que je vous aime | i just want to say i love you | i just want to say i love you | i just want to say i love you |

## 五、实验结果

### 5.1 指标汇总

| 任务 | 方向 | 数据量 | Val Loss | Val Accuracy | 备注 |
|------|------|--------|----------|-------------|------|
| **任务1: SimpleRNN** | 文本二分类 | 1000 句 | 0.6464 | **86.00%** | 长短句分类 |
| **任务2: LSTM** | TSLA 股价 | 1858 行 | — | MAE=9.92, RMSE=12.80 | 回归任务 |
| **任务3: Seq2Seq** | en→fr | 150K 对 | 1.4744 | **68.63%** | Masked Accuracy |
| **任务3: Seq2Seq** | fr→en | 150K 对 | 1.4059 | **72.20%** | Masked Accuracy |
| **任务4: Attention** | en→fr | 150K 对 | 1.2583 | **75.90%** | Masked Accuracy |
| **任务4: Attention** | fr→en | 150K 对 | 1.2268 | **79.00%** | Masked Accuracy |

### 5.2 结果分析

1. **任务1 (SimpleRNN)**：验证准确率达到 86.00%，说明即使是最简单的 RNN 结构，配合词嵌入也能很好地区分长短句。SimpleRNN 对短序列分类任务具有良好的效果。

2. **任务2 (LSTM 股票预测)**：MAE 为 9.92 美元，RMSE 为 12.80 美元。从预测曲线可以看出，模型能够较好地追踪 TSLA 股价的长期趋势，但在价格剧烈波动（如 2017 年中的快速上涨阶段）时存在滞后性。双层 LSTM 结构有效捕捉了时间序列的时序依赖。

3. **任务3 vs 任务4 对比（机器翻译）**：
   - en→fr 方向：Attention 模型比基础 Seq2Seq 提高了 **7.27 个百分点**（68.63% → 75.90%）
   - fr→en 方向：Attention 模型比基础 Seq2Seq 提高了 **6.80 个百分点**（72.20% → 79.00%）
   - 两个翻译方向的 Loss 也均有明显下降，说明 Attention 机制能够有效缓解长句翻译中的信息瓶颈问题
   - fr→en 方向的准确率整体高于 en→fr，可能与法语到英语的词序对齐更直接有关

4. **Greedy vs Beam Search**：从翻译示例来看，Beam Search (k=5) 在部分句子上生成了更流畅的译文，但在大多数简单句上与 Greedy 解码差异不大。对于复杂长句，Beam Search 的优势会更明显。

## 六、后续改进方向

1. **使用更大的预训练词向量**：当前使用随机初始化的 Embedding 层，可以替换为 GloVe、FastText 或 BPE 子词嵌入来提升翻译质量。

2. **增加数据量与清洗**：fra.txt 数据集虽然句对数量大，但部分句子存在噪声。进一步清洗和扩充语料库有助于提升模型泛化能力。

3. **尝试 Transformer 架构**：当前任务3和任务4均基于 LSTM 的 Seq2Seq 架构，可以尝试完全基于 Attention 的 Transformer 模型，其在长距离依赖建模上优于 RNN 系列。

4. **BLEU 评估**：目前仅使用 Masked Accuracy 作为评价指标，BLEU 分数作为机器翻译的行业标准评估指标，能更全面反映译文质量。

5. **超参数调优**：当前各任务的超参数（如 latent_dim、dropout、learning rate 等）仍有进一步调优空间，可通过网格搜索或贝叶斯优化寻找更优配置。

6. **增加解码策略多样性**：任务4已实现了 Beam Search，可以进一步尝试多样化的 Beam Search、长度惩罚调优等策略，减少重复生成和漏译。

7. **交互翻译界面优化**：当前交互脚本为命令行模式，可扩展为 Web 界面（如 Gradio 或 Flask），提升用户体验。

## 七、总结

本实验围绕 Keras 深度学习框架，系统地实现了四种文本处理任务：

- **SimpleRNN** 成功完成了文本二分类，验证了循环神经网络在文本特征提取上的基本能力；
- **LSTM** 在股票价格预测任务中展示了其对时间序列数据的建模能力；
- **基础 Seq2Seq** 实现了英法互译的端到端机器翻译流程；
- **Attention Encoder-Decoder** 通过引入双向编码器和注意力机制，在两个翻译方向上均取得了优于基础 Seq2Seq 的效果。

实验结果表明，Attention 机制能有效提升序列到序列模型的翻译质量，同时 Beam Search 解码在复杂句子上能够生成更合理的译文。本实验涵盖了从基础 RNN 到高级 Attention 机制的完整技术链条，为后续深入研究和工程应用奠定了基础。
