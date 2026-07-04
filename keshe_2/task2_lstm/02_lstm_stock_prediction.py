import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "stock_data.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Keras LSTM model for stock close prediction.")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stock", type=str, default="TSLA")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_stock_series(data_path, stock):
    dataframe = pd.read_csv(data_path)
    required_columns = {"Date", "Close", "Stock"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    stock_data = dataframe[dataframe["Stock"].str.upper() == stock.upper()].copy()
    if stock_data.empty:
        available = sorted(dataframe["Stock"].dropna().unique())
        raise ValueError(f"Stock {stock!r} not found. Available stocks: {available}")

    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data = stock_data.sort_values("Date").reset_index(drop=True)
    stock_data = stock_data[["Date", "Close"]].dropna()
    return stock_data


def build_sequences(values, window_size):
    features = []
    targets = []
    for index in range(window_size, len(values)):
        features.append(values[index - window_size : index])
        targets.append(values[index])
    return np.array(features), np.array(targets)


def split_by_time(features, targets, dates, train_ratio=0.8):
    split_index = int(len(features) * train_ratio)
    return (
        features[:split_index],
        features[split_index:],
        targets[:split_index],
        targets[split_index:],
        dates[:split_index],
        dates[split_index:],
    )


def build_model(window_size):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(window_size, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ],
        name="lstm_stock_close_predictor",
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def save_training_curve(history, output_dir):
    loss_path = output_dir / "lstm_loss_curve.png"

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Stock Prediction Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    return {"loss": loss_path}


def save_prediction_plot(dates, actual_values, predicted_values, stock, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual_values, label="Actual Close")
    plt.plot(dates, predicted_values, label="Predicted Close")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{stock.upper()} Close Price Prediction")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "lstm_prediction_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    stock_data = load_stock_series(args.data_path, args.stock)
    if len(stock_data) <= args.window_size + 10:
        raise ValueError("Not enough rows for the selected window size.")

    close_values = stock_data["Close"].to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(close_values)

    features, targets = build_sequences(scaled_values, args.window_size)
    target_dates = stock_data["Date"].iloc[args.window_size:].to_numpy()
    (
        train_features,
        test_features,
        train_targets,
        test_targets,
        _,
        test_dates,
    ) = split_by_time(features, targets, target_dates)

    print(f"Stock: {args.stock.upper()}")
    print(f"Rows: {len(stock_data)}")
    print(f"Date range: {stock_data['Date'].min().date()} to {stock_data['Date'].max().date()}")
    print(f"Window size: {args.window_size}")
    print(f"Train sequences: {len(train_features)}")
    print(f"Test sequences: {len(test_features)}")

    model = build_model(args.window_size)
    model.summary()

    history = model.fit(
        train_features,
        train_targets,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False,
    )

    scaled_predictions = model.predict(test_features, verbose=0)
    predictions = scaler.inverse_transform(scaled_predictions).reshape(-1)
    actual_values = scaler.inverse_transform(test_targets).reshape(-1)

    mae = mean_absolute_error(actual_values, predictions)
    rmse = mean_squared_error(actual_values, predictions) ** 0.5

    prediction_rows = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_dates).strftime("%Y-%m-%d"),
            "Actual_Close": actual_values,
            "Predicted_Close": predictions,
        }
    )
    prediction_csv = args.output_dir / "lstm_predictions.csv"
    prediction_rows.to_csv(prediction_csv, index=False, encoding="utf-8")

    curve_paths = save_training_curve(history, args.output_dir)
    prediction_plot_path = save_prediction_plot(
        pd.to_datetime(test_dates),
        actual_values,
        predictions,
        args.stock,
        args.output_dir,
    )

    metrics_path = args.output_dir / "lstm_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"tensorflow_version: {tf.__version__}",
                f"gpu_devices: {tf.config.list_physical_devices('GPU')}",
                f"stock: {args.stock.upper()}",
                f"rows: {len(stock_data)}",
                f"window_size: {args.window_size}",
                f"train_sequences: {len(train_features)}",
                f"test_sequences: {len(test_features)}",
                f"mae: {mae:.4f}",
                f"rmse: {rmse:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Saved loss curve: {curve_paths['loss']}")
    print(f"Saved prediction curve: {prediction_plot_path}")
    print(f"Saved predictions: {prediction_csv}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()

