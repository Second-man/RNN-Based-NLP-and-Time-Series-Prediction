# RNN 功能实验

本实验对应文本类项目第 1 项：基于 RNN 代码，初步实现 RNN 功能。

## 目录说明

- `01_rnn_text_sequence.py`：SimpleRNN 文本分类实验脚本。
- `outputs/`：已生成的训练曲线、预测示例和指标文件。
- 默认读取任务四中的英法 parquet 文本作为英文句子来源：`E:\keshe_2\task4_encoder_decoder_translation\data\train-00000-of-00001.parquet`。

## 运行环境

```powershell
conda activate keras_text
pip install -r E:\keshe_2\requirements.txt
```

## 快速运行

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task1_rnn\01_rnn_text_sequence.py --max_samples 2000 --epochs 2
```

## 正式小规模运行

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task1_rnn\01_rnn_text_sequence.py --max_samples 10000 --epochs 5
```

## 输出文件

- `E:\keshe_2\task1_rnn\outputs\rnn_training_curve.png`
- `E:\keshe_2\task1_rnn\outputs\rnn_predictions.txt`
- `E:\keshe_2\task1_rnn\outputs\rnn_metrics.txt`

## 成功标准

- 模型结构包含 `TextVectorization`、`Embedding`、`SimpleRNN`、`Dense`。
- 训练过程 loss 正常下降。
- 验证 accuracy 高于随机水平。
- `outputs/` 中生成曲线、预测示例和指标文件。
