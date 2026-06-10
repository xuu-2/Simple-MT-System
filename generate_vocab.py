"""
为已训练的模型重建词表文件 vocab.json
必须使用与训练时完全相同的参数，否则词表大小会不匹配。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd

UNK, PAD, SOS, EOS = '<unk>', '<pad>', '<sos>', '<eos>'


def tokenize_zh(text):
    text = re.sub(r'\s+', '', str(text).strip())
    return [ch for ch in text if ch]


def tokenize_en_simple(text):
    text = str(text).lower().strip()
    return re.findall(r"[a-zA-Z]+|\d+|[^\w\s]", text)


def build_vocab(token_lists, min_freq):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    itos = [UNK, PAD, SOS, EOS]
    seen = set(itos)
    for token, freq in counter.items():
        if freq >= min_freq and token not in seen:
            itos.append(token)
            seen.add(token)
    stoi = {token: idx for idx, token in enumerate(itos)}
    return stoi, itos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='outputs_zh_en')
    parser.add_argument('--data-csv', type=str, default='data/zh_en_opensubtitles.csv')
    parser.add_argument('--src-col', type=str, default='zh')
    parser.add_argument('--tgt-col', type=str, default='en')
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--max-train-samples', type=int, default=8000)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_path = Path(args.data_csv)

    if not data_path.exists():
        print("错误：找不到数据文件", data_path)
        return

    print("读取数据...")
    df = pd.read_csv(data_path).dropna(subset=[args.src_col, args.tgt_col])
    df[args.src_col] = df[args.src_col].astype(str).str.strip()
    df[args.tgt_col] = df[args.tgt_col].astype(str).str.strip()
    df = df[
        (df[args.src_col] != '') & (df[args.tgt_col] != '')
    ].sample(frac=1, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    test_n = max(1, int(n * args.test_ratio))
    valid_n = max(1, int(n * args.valid_ratio))
    train_df = df.iloc[: max(0, n - valid_n - test_n)]
    if args.max_train_samples:
        train_df = train_df.head(args.max_train_samples)

    print(f"使用训练集 {len(train_df)} 条构建词表...")

    src_tokens_list = []
    tgt_tokens_list = []
    for _, row in train_df.iterrows():
        src = tokenize_zh(row[args.src_col])
        tgt = tokenize_en_simple(row[args.tgt_col])
        src_tokens_list.append([SOS] + src + [EOS])
        tgt_tokens_list.append([SOS] + tgt + [EOS])

    src_stoi, src_itos = build_vocab(src_tokens_list, args.min_freq)
    trg_stoi, trg_itos = build_vocab(tgt_tokens_list, args.min_freq)

    print(f"源语言词表大小: {len(src_itos)}")
    print(f"目标语言词表大小: {len(trg_itos)}")

    output_path = model_dir / 'vocab.json'
    model_dir.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'src_stoi': src_stoi,
            'src_itos': src_itos,
            'trg_stoi': trg_stoi,
            'trg_itos': trg_itos,
        }, f, ensure_ascii=False)

    print("[OK] 词表已保存到:", output_path)


if __name__ == '__main__':
    main()
