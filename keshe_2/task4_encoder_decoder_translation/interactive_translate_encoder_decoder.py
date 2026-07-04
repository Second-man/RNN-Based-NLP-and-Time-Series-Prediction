import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "runs" / "20260701_091043_en_fr"
TOKEN_PATTERN = re.compile(r"[^a-zA-ZÀ-ÿ' ]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive translation for task4 Attention Encoder-Decoder.")
    parser.add_argument("--beam_size", type=int, default=1, help="1 uses greedy decoding; values like 3 or 5 use beam search.")
    parser.add_argument("--length_penalty", type=float, default=0.6, help="Length penalty used by beam search.")
    parser.add_argument("--no_repeat_ngram", type=int, default=2, help="Skip candidates that repeat this n-gram size; 0 disables it.")
    return parser.parse_args()


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
    model = tf.keras.models.load_model(artifact_path / "attention_model.keras", compile=False)
    source_tokenizer = load_tokenizer(artifact_path / "source_tokenizer.json")
    target_tokenizer = load_tokenizer(artifact_path / "target_tokenizer.json")
    config = json.loads((artifact_path / "config.json").read_text(encoding="utf-8"))
    model_direction = config.get("direction")
    if model_direction and model_direction != direction:
        raise ValueError(
            f"MODEL_DIR points to a {model_direction} model, but you selected {direction}.\n"
            f"Edit MODEL_DIR or choose the matching translation direction."
        )
    return model, source_tokenizer, target_tokenizer, config


def to_sequence(tokenizer, text, max_len):
    sequence = tokenizer.texts_to_sequences([clean_text(text)])
    return tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")


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


def greedy_decode(source_sequence, model, target_tokenizer, max_decoder_len):
    start_id = target_tokenizer.word_index.get("startseq")
    end_id = target_tokenizer.word_index.get("endseq")
    reverse_index = {index: word for word, index in target_tokenizer.word_index.items()}
    decoder_sequence = np.zeros((1, max_decoder_len), dtype="int32")
    decoder_sequence[0, 0] = start_id
    token_ids = []

    for position in range(1, max_decoder_len):
        probabilities = model.predict([source_sequence, decoder_sequence], verbose=0)
        sampled_id = int(np.argmax(probabilities[0, position - 1, :]))
        if sampled_id == end_id:
            break
        token_ids.append(sampled_id)
        decoder_sequence[0, position] = sampled_id

    return tokens_to_text(token_ids, reverse_index, end_id)


def beam_search_decode(source_sequence, model, target_tokenizer, max_decoder_len, beam_size, length_penalty, no_repeat_ngram):
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


def decode_sentence(text, model, source_tokenizer, target_tokenizer, config, beam_size, length_penalty, no_repeat_ngram):
    source_sequence = to_sequence(source_tokenizer, text, config["max_encoder_len"])
    max_decoder_len = config["max_decoder_len"]
    if beam_size <= 1:
        return greedy_decode(source_sequence, model, target_tokenizer, max_decoder_len)
    return beam_search_decode(
        source_sequence,
        model,
        target_tokenizer,
        max_decoder_len,
        beam_size,
        length_penalty,
        no_repeat_ngram,
    )


def main():
    args = parse_args()
    configure_stdio()
    direction = choose_direction()
    print("正在加载模型，请稍候...")
    model, source_tokenizer, target_tokenizer, config = load_artifacts(direction)
    mode = "greedy decoding" if args.beam_size <= 1 else f"beam search (beam_size={args.beam_size})"
    print(f"模型加载完成，当前解码方式: {mode}。请输入要翻译的句子，输入 q / quit / exit 退出。")
    while True:
        try:
            text = input("> ").replace("\x00", "").strip()
        except KeyboardInterrupt:
            print("\n已退出")
            break
        if text.lower() in {"q", "quit", "exit"}:
            print("已退出")
            break
        if not text:
            continue
        result = decode_sentence(
            text,
            model,
            source_tokenizer,
            target_tokenizer,
            config,
            args.beam_size,
            args.length_penalty,
            args.no_repeat_ngram,
        )
        print("翻译结果:", result if result else "(空输出，建议输入更短或更常见的句子)")


if __name__ == "__main__":
    main()
