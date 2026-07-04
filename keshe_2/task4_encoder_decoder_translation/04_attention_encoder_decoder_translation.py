import argparse
from datetime import datetime
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
#DEFAULT_DATA_PATH = BASE_DIR / "data" / "train-00000-of-00001.parquet"
DEFAULT_DATA_PATH = Path("/home/etobe/keshe_2/fra.txt")
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"
TOKEN_PATTERN = re.compile(r"[^a-zA-ZÀ-ÿ' ]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an attention Encoder-Decoder translator.")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact_dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--direction", choices=["en-fr", "fr-en"], default="fr-en")
    parser.add_argument("--max_samples", type=int, default=150000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_vocab", type=int, default=8000)
    parser.add_argument("--max_encoder_len", type=int, default=30)
    parser.add_argument("--max_decoder_len", type=int, default=35)
    parser.add_argument("--embedding_dim", type=int, default=192)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--no_repeat_ngram", type=int, default=2)
    return parser.parse_args()


def direction_dir(direction):
    return direction.replace("-", "_")


def make_run_id(direction):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{direction_dir(direction)}"


def clean_text(value):
    value = str(value).lower().strip()
    value = TOKEN_PATTERN.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_pairs(dataframe, direction, max_encoder_len=None, max_decoder_len=None):
    if "translation" in dataframe.columns:
        english = dataframe["translation"].apply(lambda item: item.get("en") if isinstance(item, dict) else None)
        french = dataframe["translation"].apply(lambda item: item.get("fr") if isinstance(item, dict) else None)
    elif {"en", "fr"}.issubset(dataframe.columns):
        english = dataframe["en"]
        french = dataframe["fr"]
    else:
        raise ValueError("Expected a parquet translation column or txt-style en/fr columns.")

    if direction == "en-fr":
        source, target = english, french
    else:
        source, target = french, english

    pairs = pd.DataFrame({"source": source, "target": target}).dropna()
    pairs["source"] = pairs["source"].map(clean_text)
    pairs["target"] = pairs["target"].map(clean_text)
    source_lengths = pairs["source"].str.split().str.len()
    target_lengths = pairs["target"].str.split().str.len()
    pairs = pairs[(source_lengths >= 2) & (target_lengths >= 2)]
    if max_encoder_len:
        pairs = pairs[pairs["source"].str.split().str.len() <= max_encoder_len]
    if max_decoder_len:
        pairs = pairs[pairs["target"].str.split().str.len() <= max_decoder_len - 1]
    return pairs.drop_duplicates().reset_index(drop=True)


def load_translation_pairs(data_path, max_samples, seed, direction, max_encoder_len, max_decoder_len):
    if data_path.suffix.lower() == ".parquet":
        dataframe = pd.read_parquet(data_path)
    elif data_path.suffix.lower() == ".txt":
        dataframe = pd.read_csv(data_path, sep="	", header=None, usecols=[0, 1], names=["en", "fr"], dtype=str)
    else:
        raise ValueError(f"Unsupported data file format: {data_path.suffix}. Use .parquet or .txt.")

    pairs = extract_pairs(dataframe, direction, max_encoder_len, max_decoder_len)
    if max_samples and len(pairs) > max_samples:
        pairs = pairs.sample(max_samples, random_state=seed)
    pairs = pairs.reset_index(drop=True)
    if len(pairs) < 500:
        raise ValueError(f"Not enough sentence pairs after cleaning: {len(pairs)}")
    pairs["decoder_input"] = "startseq " + pairs["target"]
    pairs["decoder_target"] = pairs["target"] + " endseq"
    return pairs

def make_tokenizer(texts, max_vocab):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab, filters="", oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def to_padded_sequences(tokenizer, texts, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")



@tf.keras.utils.register_keras_serializable()
class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, ignore_token_ids=None, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_token_ids = [int(token_id) for token_id in (ignore_token_ids or []) if token_id is not None]
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.cast(y_true, tf.int64), axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int64)

        mask = tf.not_equal(y_true, 0)
        for token_id in self.ignore_token_ids:
            mask = tf.logical_and(mask, tf.not_equal(y_true, tf.cast(token_id, tf.int64)))

        matches = tf.logical_and(tf.equal(y_true, y_pred), mask)
        self.correct.assign_add(tf.reduce_sum(tf.cast(matches, tf.float32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"ignore_token_ids": self.ignore_token_ids})
        return config

def build_attention_model(src_vocab_size, tgt_vocab_size, max_encoder_len, max_decoder_len, embedding_dim, latent_dim, target_unk_id):
    encoder_inputs = tf.keras.Input(shape=(max_encoder_len,), name="encoder_inputs")
    encoder_embedding = tf.keras.layers.Embedding(src_vocab_size, embedding_dim, mask_zero=True, name="encoder_embedding")(
        encoder_inputs
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.3),
        name="bidirectional_encoder_lstm",
    )(encoder_embedding)
    state_h = tf.keras.layers.Concatenate(name="encoder_state_h")([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate(name="encoder_state_c")([forward_c, backward_c])
    decoder_units = latent_dim * 2

    decoder_inputs = tf.keras.Input(shape=(max_decoder_len,), name="decoder_inputs")
    decoder_embedding = tf.keras.layers.Embedding(tgt_vocab_size, embedding_dim, mask_zero=True, name="decoder_embedding")(
        decoder_inputs
    )
    decoder_outputs = tf.keras.layers.LSTM(decoder_units, return_sequences=True, name="decoder_lstm",dropout=0.3)(
        decoder_embedding,
        initial_state=[state_h, state_c],
    )
    context = tf.keras.layers.Attention(name="attention")([decoder_outputs, encoder_outputs])
    decoder_context = tf.keras.layers.Concatenate(name="decoder_context")([decoder_outputs, context])
    decoder_context = tf.keras.layers.Dropout(0.2, name="decoder_dropout")(decoder_context)
    outputs = tf.keras.layers.Dense(tgt_vocab_size, activation="softmax", name="decoder_softmax")(decoder_context)
    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs, name="attention_encoder_decoder_translation")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[MaskedAccuracy([target_unk_id])])
    return model


def greedy_decode(model, source_sequence, target_tokenizer, max_decoder_len):
    start_id = target_tokenizer.word_index.get("startseq")
    end_id = target_tokenizer.word_index.get("endseq")
    reverse_index = {index: word for word, index in target_tokenizer.word_index.items()}
    decoder_sequence = np.zeros((1, max_decoder_len), dtype="int32")
    decoder_sequence[0, 0] = start_id
    decoded_words = []
    for position in range(1, max_decoder_len):
        probabilities = model.predict([source_sequence, decoder_sequence], verbose=0)
        sampled_id = int(np.argmax(probabilities[0, position - 1, :]))
        sampled_word = reverse_index.get(sampled_id, "<unk>")
        if sampled_id == end_id or sampled_word == "endseq":
            break
        if sampled_word not in ("startseq", "<unk>"):
            decoded_words.append(sampled_word)
        decoder_sequence[0, position] = sampled_id
    return " ".join(decoded_words)


def has_repeated_ngram(token_ids, ngram_size):
    if ngram_size <= 0 or len(token_ids) < ngram_size:
        return False
    ngrams = [tuple(token_ids[index : index + ngram_size]) for index in range(len(token_ids) - ngram_size + 1)]
    return len(ngrams) != len(set(ngrams))


def sequence_score(log_probability, token_count, length_penalty):
    token_count = max(token_count, 1)
    if length_penalty <= 0:
        return log_probability
    return log_probability / (token_count ** length_penalty)


def tokens_to_text(token_ids, reverse_index, end_id):
    words = []
    for token_id in token_ids:
        word = reverse_index.get(int(token_id), "<unk>")
        if token_id == end_id or word == "endseq":
            break
        if word not in {"startseq", "<unk>"}:
            words.append(word)
    return " ".join(words).strip()


def beam_search_decode(model, source_sequence, target_tokenizer, max_decoder_len, beam_size, length_penalty, no_repeat_ngram):
    start_id = target_tokenizer.word_index.get("startseq")
    end_id = target_tokenizer.word_index.get("endseq")
    reverse_index = {index: word for word, index in target_tokenizer.word_index.items()}
    if not start_id or not end_id:
        return ""

    beams = [([start_id], 0.0, False)]
    completed = []
    epsilon = 1e-9

    for position in range(1, max_decoder_len):
        candidates = []
        for token_ids, log_probability, finished in beams:
            if finished:
                completed.append((token_ids, log_probability, True))
                continue

            decoder_sequence = np.zeros((1, max_decoder_len), dtype="int32")
            decoder_sequence[0, : min(len(token_ids), max_decoder_len)] = token_ids[:max_decoder_len]
            probabilities = model.predict([source_sequence, decoder_sequence], verbose=0)[0, position - 1, :]
            top_ids = np.argsort(probabilities)[-beam_size * 3 :][::-1]

            for sampled_id in top_ids:
                sampled_id = int(sampled_id)
                new_tokens = token_ids + [sampled_id]
                if has_repeated_ngram(new_tokens[1:], no_repeat_ngram):
                    continue
                new_log_probability = log_probability + float(np.log(probabilities[sampled_id] + epsilon))
                candidates.append((new_tokens, new_log_probability, sampled_id == end_id))

        if not candidates:
            break

        candidates.sort(
            key=lambda item: sequence_score(item[1], len(item[0]) - 1, length_penalty),
            reverse=True,
        )
        beams = candidates[:beam_size]
        completed.extend([item for item in beams if item[2]])
        if beams and all(item[2] for item in beams):
            break

    final_candidates = completed or beams
    final_candidates.sort(
        key=lambda item: sequence_score(item[1], len(item[0]) - 1, length_penalty),
        reverse=True,
    )
    best_tokens = final_candidates[0][0][1:]
    return tokens_to_text(best_tokens, reverse_index, end_id)


def save_tokenizer(tokenizer, path):
    path.write_text(tokenizer.to_json(), encoding="utf-8")


def write_artifacts(artifact_path, model, source_tokenizer, target_tokenizer, config):
    artifact_path.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path / "attention_model.keras")
    save_tokenizer(source_tokenizer, artifact_path / "source_tokenizer.json")
    save_tokenizer(target_tokenizer, artifact_path / "target_tokenizer.json")
    (artifact_path / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def save_artifacts(artifact_root, run_id, model, source_tokenizer, target_tokenizer, config):
    run_artifact_path = artifact_root / "runs" / run_id
    write_artifacts(run_artifact_path, model, source_tokenizer, target_tokenizer, config)
    return run_artifact_path


def save_training_curve(history, output_dir, direction, run_id=None):
    metric_name = "masked_accuracy" if "masked_accuracy" in history.history else "accuracy"
    val_metric_name = f"val_{metric_name}"

    if run_id:
        output_path = output_dir / "runs" / run_id
    else:
        output_path = output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    accuracy_path = output_path / "encoder_decoder_accuracy_curve.png"
    loss_path = output_path / "encoder_decoder_loss_curve.png"

    plt.figure(figsize=(8, 4))
    if metric_name in history.history:
        plt.plot(history.history[metric_name], label="train_masked_acc")
    if val_metric_name in history.history:
        plt.plot(history.history[val_metric_name], label="val_masked_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Attention Encoder-Decoder Accuracy ({direction})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Attention Encoder-Decoder Loss ({direction})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    return {"accuracy": accuracy_path, "loss": loss_path}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or make_run_id(args.direction)
    run_output_dir = args.output_dir / "runs" / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Direction:", args.direction)

    pairs = load_translation_pairs(args.data_path, args.max_samples, args.seed, args.direction, args.max_encoder_len, args.max_decoder_len)
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.1, random_state=args.seed)
    source_tokenizer = make_tokenizer(train_pairs["source"], args.max_vocab)
    target_tokenizer = make_tokenizer(pd.concat([train_pairs["decoder_input"], train_pairs["decoder_target"]]), args.max_vocab)

    encoder_input_data = to_padded_sequences(source_tokenizer, train_pairs["source"], args.max_encoder_len)
    decoder_input_data = to_padded_sequences(target_tokenizer, train_pairs["decoder_input"], args.max_decoder_len)
    decoder_target_data = np.expand_dims(to_padded_sequences(target_tokenizer, train_pairs["decoder_target"], args.max_decoder_len), -1)
    val_encoder_input = to_padded_sequences(source_tokenizer, test_pairs["source"], args.max_encoder_len)
    val_decoder_input = to_padded_sequences(target_tokenizer, test_pairs["decoder_input"], args.max_decoder_len)
    val_decoder_target = np.expand_dims(to_padded_sequences(target_tokenizer, test_pairs["decoder_target"], args.max_decoder_len), -1)

    src_vocab_size = min(args.max_vocab, len(source_tokenizer.word_index) + 1)
    tgt_vocab_size = min(args.max_vocab, len(target_tokenizer.word_index) + 1)
    target_unk_id = target_tokenizer.word_index.get("<unk>")
    print(f"Sentence pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(test_pairs)}")
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")

    model = build_attention_model(src_vocab_size, tgt_vocab_size, args.max_encoder_len, args.max_decoder_len, args.embedding_dim, args.latent_dim, target_unk_id)
    model.summary()
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        validation_data=([val_encoder_input, val_decoder_input], val_decoder_target),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    val_loss, val_masked_accuracy = model.evaluate([val_encoder_input, val_decoder_input], val_decoder_target, verbose=0)

    config = {
        "direction": args.direction,
        "max_encoder_len": args.max_encoder_len,
        "max_decoder_len": args.max_decoder_len,
        "max_vocab": args.max_vocab,
        "source_vocab_size": src_vocab_size,
        "target_vocab_size": tgt_vocab_size,
        "model": "bidirectional_lstm_encoder_attention_decoder",
        "run_id": run_id,
        "data_path": str(args.data_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "embedding_dim": args.embedding_dim,
        "latent_dim": args.latent_dim,
        "seed": args.seed,
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "no_repeat_ngram": args.no_repeat_ngram,
    }
    artifact_path = save_artifacts(args.artifact_dir, run_id, model, source_tokenizer, target_tokenizer, config)

    example_rows = test_pairs.head(10).reset_index(drop=True)
    example_source_sequences = to_padded_sequences(source_tokenizer, example_rows["source"], args.max_encoder_len)
    lines = [f"Attention Encoder-Decoder translation examples ({args.direction})", ""]
    for index, row in example_rows.iterrows():
        source_sequence = example_source_sequences[index : index + 1]
        greedy_predicted = greedy_decode(model, source_sequence, target_tokenizer, args.max_decoder_len)
        lines.append(f"source: {row['source']}")
        lines.append(f"target: {row['target']}")
        lines.append(f"greedy_predicted: {greedy_predicted}")
        if args.beam_size > 1:
            beam_predicted = beam_search_decode(
                model,
                source_sequence,
                target_tokenizer,
                args.max_decoder_len,
                args.beam_size,
                args.length_penalty,
                args.no_repeat_ngram,
            )
            lines.append(f"beam_predicted: {beam_predicted}")
        lines.append("")

    examples_path = run_output_dir / "encoder_decoder_examples.txt"
    examples_path.write_text("\n".join(lines), encoding="utf-8")
    curve_paths = save_training_curve(history, args.output_dir, args.direction, run_id)
    metrics_path = run_output_dir / "encoder_decoder_metrics.txt"
    metrics_path.write_text(
        "\n".join([
            f"tensorflow_version: {tf.__version__}",
            f"run_id: {run_id}",
            f"data_path: {args.data_path}",
            f"epochs: {args.epochs}",
            f"batch_size: {args.batch_size}",
            f"max_samples: {args.max_samples}",
            f"max_vocab: {args.max_vocab}",
            f"embedding_dim: {args.embedding_dim}",
            f"latent_dim: {args.latent_dim}",
            f"seed: {args.seed}",
            f"beam_size: {args.beam_size}",
            f"length_penalty: {args.length_penalty}",
            f"no_repeat_ngram: {args.no_repeat_ngram}",
            f"gpu_devices: {tf.config.list_physical_devices('GPU')}",
            f"direction: {args.direction}",
            f"sample_count: {len(pairs)}",
            f"train_pairs: {len(train_pairs)}",
            f"validation_pairs: {len(test_pairs)}",
            f"source_vocab_size: {src_vocab_size}",
            f"target_vocab_size: {tgt_vocab_size}",
            f"max_encoder_len: {args.max_encoder_len}",
            f"max_decoder_len: {args.max_decoder_len}",
            "model: bidirectional_lstm_encoder_attention_decoder",
            f"validation_loss: {val_loss:.4f}",
            f"validation_masked_accuracy: {val_masked_accuracy:.4f}",
            f"artifact_path: {artifact_path}",
        ]),
        encoding="utf-8",
    )
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation masked accuracy: {val_masked_accuracy:.4f}")
    print(f"Run ID: {run_id}")
    print(f"Saved artifacts: {artifact_path}")
    print(f"Saved accuracy curve: {curve_paths['accuracy']}")
    print(f"Saved loss curve: {curve_paths['loss']}")
    print(f"Saved examples: {examples_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
