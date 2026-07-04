# 编码-解码机器翻译实验

本目录对应文本类项目第 4 项：基于编解码代码，实现机器翻译。

## 数据集

- `data/train-00000-of-00001.parquet`：英法翻译文本数据。
- 数据列为 `id` 和 `translation`，其中 `translation` 包含 `en` 和 `fr` 句对。

## 方法

本任务实现比任务 3 更明确的 Encoder-Decoder 改进版本：

```text
Encoder: source tokens -> Embedding -> Bidirectional LSTM
Decoder: target start tokens -> Embedding -> LSTM
Attention: Decoder outputs attend to Encoder outputs
Output: Attention context + Decoder outputs -> Dense Softmax
```

相比任务 3 的基础 Seq2Seq，本任务增加了双向 LSTM Encoder 和 Attention 层。

## 训练并保存模型

英文到法语：

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task4_encoder_decoder_translation\04_attention_encoder_decoder_translation.py --direction en-fr --max_samples 8000 --epochs 8
```

法语到英文：

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task4_encoder_decoder_translation\04_attention_encoder_decoder_translation.py --direction fr-en --max_samples 8000 --epochs 8
```

保存目录：

- `artifacts/en_fr/`
- `artifacts/fr_en/`

## 终端交互翻译

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task4_encoder_decoder_translation\interactive_translate_encoder_decoder.py
```

启动后选择翻译方向，然后反复输入句子。输入 `q`、`quit` 或 `exit` 退出。

## 输出文件

- `outputs/encoder_decoder_training_curve_en_fr.png`
- `outputs/encoder_decoder_examples_en_fr.txt`
- `outputs/encoder_decoder_metrics_en_fr.txt`

## 成功标准

- 模型结构包含 `Bidirectional LSTM`、`Decoder LSTM` 和 `Attention`。
- 能读取英法句对并完成训练。
- 能保存完整模型和 tokenizer。
- 终端交互脚本能加载模型并持续翻译输入句子。
- 可与任务 3 的基础 Seq2Seq 结果对比。
