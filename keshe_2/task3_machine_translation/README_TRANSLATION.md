# 机器翻译实验

本目录对应文本类项目第 3 项：基于英法互译数据和代码，实现机器翻译。

## 数据集

默认复用任务四中的英法 parquet 数据：

`E:\keshe_2\task4_encoder_decoder_translation\data\train-00000-of-00001.parquet`

数据列为 `id` 和 `translation`，其中 `translation` 包含 `en` 和 `fr` 句对。

## 方法

使用基础词级 Seq2Seq LSTM：

```text
Encoder: source tokens -> Embedding -> LSTM states
Decoder: target start tokens -> Embedding -> LSTM -> Dense Softmax
```

训练时使用 teacher forcing，推理时从 `startseq` 开始逐词生成。

## 训练并保存模型

英文到法语：

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task3_machine_translation\03_seq2seq_translation.py --direction en-fr --max_samples 8000 --epochs 8
```

法语到英文：

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task3_machine_translation\03_seq2seq_translation.py --direction fr-en --max_samples 8000 --epochs 8
```

保存目录：

- `artifacts/en_fr/`
- `artifacts/fr_en/`

## 终端交互翻译

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task3_machine_translation\interactive_translate_seq2seq.py
```

启动后选择翻译方向，然后反复输入句子。输入 `q`、`quit` 或 `exit` 退出。

## 输出文件

- `outputs/translation_training_curve_en_fr.png`
- `outputs/translation_examples_en_fr.txt`
- `outputs/translation_metrics_en_fr.txt`

## 成功标准

- 模型结构包含 Encoder LSTM 和 Decoder LSTM。
- 能读取英法句对并完成训练。
- 能保存模型和 tokenizer。
- 终端交互脚本能加载模型并持续翻译输入句子。
