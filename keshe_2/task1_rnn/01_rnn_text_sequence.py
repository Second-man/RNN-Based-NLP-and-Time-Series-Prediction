import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_DATA_PATH = (
    PROJECT_DIR
    / "task4_encoder_decoder_translation"
    / "data"
    / "train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small Keras SimpleRNN model on English text length labels."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_english_texts(dataframe):
    if "translation" in dataframe.columns:
        translations = dataframe["translation"]
        first_valid = translations.dropna().iloc[0]
        if isinstance(first_valid, dict):
            return translations.apply(lambda item: item.get("en") if isinstance(item, dict) else None)
        raise ValueError("Found a translation column, but its values are not dictionaries.")

    for column_name in ("en", "eng", "english"):
        if column_name in dataframe.columns:
            return dataframe[column_name]

    raise ValueError(
        "Could not find English text. Expected a translation column or one of: en, eng, english."
    )


def load_texts(data_path, max_samples):
    dataframe = pd.read_parquet(data_path)
    texts = extract_english_texts(dataframe)
    texts = texts.dropna().astype(str).str.strip()
    texts = texts[texts.str.split().str.len() >= 3]

    if max_samples and len(texts) > max_samples:
        texts = texts.sample(max_samples, random_state=42)

    texts = texts.reset_index(drop=True)
    if len(texts) < 100:
        raise ValueError(f"Not enough usable English texts: {len(texts)}")
    return texts


def build_length_labels(texts):
    word_counts = texts.str.split().str.len()
    threshold = int(word_counts.median())
    labels = (word_counts > threshold).astype("int32").to_numpy()

    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() == 0:
        raise ValueError("Only one class was created. Please use more varied text samples.")
    return labels, threshold, class_counts


def build_model(max_tokens, sequence_length):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype=tf.string),
            vectorizer,
            tf.keras.layers.Embedding(max_tokens, 64, mask_zero=True),
            tf.keras.layers.SimpleRNN(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="simple_rnn_text_length_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, vectorizer


def save_training_curve(history, output_dir):
    accuracy_path = output_dir / "rnn_accuracy_curve.png"
    loss_path = output_dir / "rnn_loss_curve.png"

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SimpleRNN Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SimpleRNN Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    return {"accuracy": accuracy_path, "loss": loss_path}


def save_predictions(model, sample_texts, threshold, output_dir):
    probabilities = model.predict(sample_texts.to_numpy(), verbose=0).reshape(-1)
    lines = [
        "SimpleRNN prediction examples",
        f"Label rule: word_count > {threshold} => long sentence (1), otherwise short sentence (0)",
        "",
    ]

    for text, probability in zip(sample_texts, probabilities):
        predicted_label = int(probability >= 0.5)
        word_count = len(text.split())
        lines.append(f"text: {text}")
        lines.append(f"word_count: {word_count}")
        lines.append(f"predicted_probability_long: {probability:.4f}")
        lines.append(f"predicted_label: {predicted_label}")
        lines.append("")

    prediction_path = output_dir / "rnn_predictions.txt"
    prediction_path.write_text("\n".join(lines), encoding="utf-8")
    return prediction_path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    texts = load_texts(args.data_path, args.max_samples)
    labels, threshold, class_counts = build_length_labels(texts)
    print(f"Loaded texts: {len(texts)}")
    print(f"Length threshold: {threshold} words")
    print(f"Class counts [short, long]: {class_counts.tolist()}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    model, vectorizer = build_model(args.max_tokens, args.sequence_length)
    vectorizer.adapt(train_texts.to_numpy())
    model.summary()

    history = model.fit(
        train_texts.to_numpy(),
        train_labels,
        validation_data=(val_texts.to_numpy(), val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    val_loss, val_accuracy = model.evaluate(val_texts.to_numpy(), val_labels, verbose=0)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    curve_paths = save_training_curve(history, args.output_dir)
    prediction_path = save_predictions(model, val_texts.head(50), threshold, args.output_dir)

    metrics_path = args.output_dir / "rnn_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"tensorflow_version: {tf.__version__}",
                f"gpu_devices: {tf.config.list_physical_devices('GPU')}",
                f"sample_count: {len(texts)}",
                f"length_threshold_words: {threshold}",
                f"class_counts_short_long: {class_counts.tolist()}",
                f"validation_loss: {val_loss:.4f}",
                f"validation_accuracy: {val_accuracy:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved accuracy curve: {curve_paths['accuracy']}")
    print(f"Saved loss curve: {curve_paths['loss']}")
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()


