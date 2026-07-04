# 基于 RNN 的 NLP 与时序预测

使用 TensorFlow/Keras 实现四个渐进式任务，覆盖文本分类、股价预测、Seq2Seq 翻译与注意力机制。

## 目录结构

```
keshe_2/
├── fra.txt                              # 英法平行语料 (~24 万句对, Tatoeba/Anki)
├── requirements.txt                     # Python 依赖
├── experiment_report.md                 # 实验报告
├── accuracy_log.md                      # 超参数调优记录
│
├── task1_rnn/                           # 任务1: SimpleRNN 文本分类
│   ├── 01_rnn_text_sequence.py
│   ├── README_RNN.md
│   └── outputs/
│
├── task2_lstm/                          # 任务2: LSTM 股价预测
│   ├── 02_lstm_stock_prediction.py
│   ├── README_LSTM.md
│   ├── data/stock_data.csv
│   └── outputs/
│
├── task3_machine_translation/           # 任务3: Seq2Seq 翻译
│   ├── 03_seq2seq_translation.py
│   ├── interactive_translate_seq2seq.py
│   ├── README_TRANSLATION.md
│   ├── artifacts/runs/                  # 训练好的模型与 tokenizer
│   └── outputs/runs/
│
└── task4_encoder_decoder_translation/   # 任务4: 注意力 Encoder-Decoder 翻译
    ├── 04_attention_encoder_decoder_translation.py
    ├── interactive_translate_encoder_decoder.py
    ├── README_ENCODER_DECODER.md
    ├── data/train-00000-of-00001.parquet
    ├── artifacts/runs/                  # 训练好的模型与 tokenizer
    └── outputs/runs/
```

## 环境配置

```bash
conda create -n keras_text python=3.10
conda activate keras_text
pip install -r requirements.txt
```

**依赖：**

| 包 | 版本 | 用途 |
|---|---|---|
| tensorflow | 2.21.0 | 深度学习框架 |
| pandas | * | 数据加载 |
| pyarrow | * | Parquet 读取 |
| scikit-learn | * | 数据预处理与评估 |
| matplotlib | * | 可视化 |

可选：安装 CUDA 与 cuDNN 以启用 GPU 加速。

## 各任务说明

### 任务1 — SimpleRNN 文本分类

二分类：根据英语句子的长度（词数是否超过中位数）判断"长句"或"短句"。

```bash
python task1_rnn/01_rnn_text_sequence.py --max_samples 1000 --epochs 10
```

**架构：** `TextVectorization -> Embedding -> SimpleRNN -> Dropout -> Dense(sigmoid)`

**最佳结果：** 验证准确率 **86.00%**

### 任务2 — LSTM 股价预测

基于过去 60 个交易日的 OHLCV 数据，预测下一个交易日的收盘价（回归任务）。

```bash
python task2_lstm/02_lstm_stock_prediction.py --stock TSLA --window_size 60 --epochs 30
```

**架构：** `LSTM(64) -> Dropout -> LSTM(32) -> Dense(1)`

**最佳结果 (TSLA)：** MAE = 9.92, RMSE = 12.80

### 任务3 — 基础 Seq2Seq 翻译

基于 LSTM 的词级 Seq2Seq 实现英法双向翻译。

```bash
# 英 -> 法
python task3_machine_translation/03_seq2seq_translation.py --direction en-fr --max_samples 150000 --epochs 25

# 法 -> 英
python task3_machine_translation/03_seq2seq_translation.py --direction fr-en --max_samples 150000 --epochs 25

# 交互式翻译
python task3_machine_translation/interactive_translate_seq2seq.py
```

**架构：** `LSTM Encoder -> LSTM Decoder (Teacher Forcing)`

**最佳结果：**

| 方向 | 验证损失 | Masked Accuracy |
|---|---|---|
| en → fr | 1.4744 | **68.63%** |
| fr → en | 1.4059 | **72.20%** |

### 任务4 — 注意力 Encoder-Decoder 翻译

在 Task 3 基础上加入**双向 LSTM 编码器**与**注意力机制**，支持贪婪解码与束搜索（Beam Search）。

```bash
# 英 -> 法
python task4_encoder_decoder_translation/04_attention_encoder_decoder_translation.py --direction en-fr --max_samples 150000 --epochs 25

# 交互式翻译（支持束搜索）
python task4_encoder_decoder_translation/interactive_translate_encoder_decoder.py --beam_size 5
```

**架构：** `Bidirectional LSTM Encoder -> LSTM Decoder + Attention + Beam Search`

**最佳结果：**

| 方向 | 验证损失 | Masked Accuracy | 较 Task 3 提升 |
|---|---|---|---|
| en → fr | 1.2583 | **75.90%** | +7.27pp |
| fr → en | 1.2268 | **79.00%** | +6.80pp |

## 命令行参数

所有脚本共享以下常用参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--data_path` | 语料文件路径 | 各任务不同（需按实际路径调整） |
| `--output_dir` | 输出目录 | `outputs/runs/` |
| `--max_samples` | 使用的最大样本数 | 1000~150000 |
| `--epochs` | 训练轮数 | 10~25 |
| `--batch_size` | 批次大小 | 64 |
| `--seed` | 随机种子 | 42 |

> **注意：** 代码中的默认 `data_path` 为 Linux 路径（`/home/etobe/keshe_2/fra.txt`），在 Windows 上运行时请通过 `--data_path` 参数覆盖。

## 结果总览

| 任务 | 模型 | 指标 | 最佳值 |
|---|---|---|---|
| 1 | SimpleRNN | 验证准确率 | **86.00%** |
| 2 | LSTM | MAE / RMSE | **9.92 / 12.80** |
| 3 | Seq2Seq (en→fr) | Masked Accuracy | **68.63%** |
| 3 | Seq2Seq (fr→en) | Masked Accuracy | **72.20%** |
| 4 | Attn Enc-Dec (en→fr) | Masked Accuracy | **75.90%** |
| 4 | Attn Enc-Dec (fr→en) | Masked Accuracy | **79.00%** |

## 特性

- **渐进式设计：** SimpleRNN → LSTM → Seq2Seq → Attention，逐步深入
- **自定义 MaskedAccuracy 指标：** 翻译任务中忽略 padding 和 `<unk>` token，评估更真实
- **束搜索推理：** Task 4 支持 beam search（可配束宽、长度惩罚、n-gram 去重）
- **早停机制：** Task 3 使用 `EarlyStopping` 防止过拟合
- **交互式翻译：** Task 3/4 提供 CLI 实时翻译工具
- **运行 ID 系统：** 基于时间戳组织每次训练的产物与输出

## 数据下载

运行前需将以下数据集下载并放置到对应路径：

| 文件 | 放置路径 | 下载地址 |
|---|---|---|
| 英法平行语料 | `fra.txt`（项目根目录） | [eng-fra.txt](https://github.com/L1aoXingyu/seq2seq-translation/blob/master/data/eng-fra.txt) |
| HF Parquet 数据 | `task4_encoder_decoder_translation/data/train-00000-of-00001.parquet` | [opus_books en-fr](https://gitcode.com/mirrors/Helsinki-NLP/opus_books/blob/main/en-fr/train-00000-of-00001.parquet) |
