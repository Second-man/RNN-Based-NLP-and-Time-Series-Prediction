# LSTM 股票预测实验

本目录对应文本类项目第 2 项：基于 LSTM 代码，实现 LSTM 功能。

## 数据集

- `data/stock_data.csv`：LSTM 股票预测任务数据集。
- 字段包括：`Date`、`Open`、`High`、`Low`、`Close`、`Volume`、`OpenInt`、`Stock`。
- 默认使用 `FB` 的 `Close` 字段构造时间序列窗口，训练速度快，预测曲线更适合课堂展示。

## 运行

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task2_lstm\02_lstm_stock_prediction.py --stock FB --window_size 30 --epochs 10
```

快速测试：

```powershell
D:\SYH\anaconda3\envs\keras_text\python.exe E:\keshe_2\task2_lstm\02_lstm_stock_prediction.py --stock FB --window_size 30 --epochs 2
```

## 输出文件

- `outputs/lstm_training_curve.png`
- `outputs/lstm_prediction_curve.png`
- `outputs/lstm_predictions.csv`
- `outputs/lstm_metrics.txt`

## 成功标准

- 模型结构包含两层 `LSTM` 和 `Dense` 输出层。
- 能读取 `stock_data.csv` 并按时间顺序构造滑动窗口。
- 输出 MAE、RMSE，并生成真实值/预测值对比曲线。

