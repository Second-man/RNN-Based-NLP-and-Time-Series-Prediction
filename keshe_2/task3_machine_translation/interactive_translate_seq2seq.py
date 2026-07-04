import json
import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "runs" / "20260701_102533_en_fr"
TOKEN_PATTERN = re.compile(r"[^a-zA-ZÀ-ÿ' ]+")


def configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def clean_text(value):
    value = str(value).lower().strip()
    value = TOKEN_PATTERN.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def direction_dir(direction):
    return direction.replace("-", "_")


def choose_direction():
    print("请选择翻译方向:")
    print("1. 英文 -> 法语")
    print("2. 法语 -> 英文")
    while True:
        choice = input("请输入 1 或 2: ").replace("\x00", "").strip()
        if "1" in choice:
            return "en-fr"
        if "2" in choice:
            return "fr-en"
        print("输入无效，请输入 1 或 2。")


def load_tokenizer(path):
    return tokenizer_from_json(path.read_text(encoding="utf-8"))


def load_artifacts(direction):
    artifact_path = MODEL_DIR
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {artifact_path}\n"
            f"Train a model first, or edit MODEL_DIR at the top of this script to artifacts/runs/<run_id>."
        )
    encoder_model = tf.keras.models.load_model(artifact_path / "encoder_model.keras")
    decoder_model = tf.keras.models.load_model(artifact_path / "decoder_model.keras")
    source_tokenizer = load_tokenizer(artifact_path / "source_tokenizer.json")
    target_tokenizer = load_tokenizer(artifact_path / "target_tokenizer.json")
    config = json.loads((artifact_path / "config.json").read_text(encoding="utf-8"))
    model_direction = config.get("direction")
    if model_direction and model_direction != direction:
        raise ValueError(
            f"MODEL_DIR points to a {model_direction} model, but you selected {direction}.\n"
            f"Edit MODEL_DIR or choose the matching translation direction."
        )
    return encoder_model, decoder_model, source_tokenizer, target_tokenizer, config


def to_sequence(tokenizer, text, max_len):
    sequence = tokenizer.texts_to_sequences([clean_text(text)])
    return tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")


def decode_sentence(text, encoder_model, decoder_model, source_tokenizer, target_tokenizer, config):
    input_sequence = to_sequence(source_tokenizer, text, config["max_encoder_len"])
    states = encoder_model.predict(input_sequence, verbose=0)
    start_id = target_tokenizer.word_index.get("startseq")
    end_id = target_tokenizer.word_index.get("endseq")
    target_token = np.array([[start_id]])
    reverse_index = {index: word for word, index in target_tokenizer.word_index.items()}
    decoded_words = []
    for _ in range(config["max_decoder_len"]):
        output_tokens, h, c = decoder_model.predict([target_token] + states, verbose=0)
        sampled_id = int(np.argmax(output_tokens[0, -1, :]))
        sampled_word = reverse_index.get(sampled_id, "<unk>")
        if sampled_id == end_id or sampled_word == "endseq":
            break
        if sampled_word not in ("startseq", "<unk>"):
            decoded_words.append(sampled_word)
        target_token = np.array([[sampled_id]])
        states = [h, c]
    return " ".join(decoded_words).strip()


def main():
    configure_stdio()
    direction = choose_direction()
    print("正在加载模型，请稍候...")
    encoder_model, decoder_model, source_tokenizer, target_tokenizer, config = load_artifacts(direction)
    print("模型加载完成。请输入要翻译的句子，输入 q / quit / exit 退出。")
    while True:
        text = input("> ").replace("\x00", "").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("已退出")
            break
        if not text:
            continue
        result = decode_sentence(text, encoder_model, decoder_model, source_tokenizer, target_tokenizer, config)
        print("翻译结果:", result if result else "(空输出，建议输入更短或更常见的句子)")


if __name__ == "__main__":
    main()




