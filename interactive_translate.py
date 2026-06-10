"""
交互式机器翻译脚本 - 支持 Beam Search
"""
import argparse
import json
import re
import torch
import torch.nn as nn
from pathlib import Path
import heapq

UNK, PAD, SOS, EOS = '<unk>', '<pad>', '<sos>', '<eos>'


def tokenize_zh(text):
    text = re.sub(r'\s+', '', str(text).strip())
    return [ch for ch in text if ch]


def tokenize_en_simple(text):
    text = str(text).lower().strip()
    return re.findall(r"[a-zA-Z]+|\d+|[^\w\s]", text)


class Vocab:
    def __init__(self, stoi, itos, default_index=0):
        self.stoi = stoi
        self.itos = itos
        self.default_index = default_index

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def set_default_index(self, idx):
        self.default_index = idx

    def lookup_indices(self, tokens):
        return [self.stoi.get(token, self.default_index) for token in tokens]

    def get_itos(self):
        return self.itos


class Attention(nn.Module):
    def __init__(self, eh, dh):
        super().__init__()
        self.attn = nn.Linear(eh * 2 + dh, dh)
        self.v = nn.Linear(dh, 1, bias=False)

    def forward(self, hidden, enc_out, mask):
        sl = enc_out.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, sl, 1)
        enc = enc_out.permute(1, 0, 2)
        e = torch.tanh(self.attn(torch.cat((hidden, enc), dim=2)))
        a = self.v(e).squeeze(2)
        return torch.softmax(a.masked_fill(mask == 0, -1e10), dim=1)


class Encoder(nn.Module):
    def __init__(self, inp, emb, eh, dh, drop):
        super().__init__()
        self.emb = nn.Embedding(inp, emb)
        self.rnn = nn.GRU(emb, eh, bidirectional=True)
        self.fc = nn.Linear(eh * 2, dh)
        self.drop = nn.Dropout(drop)

    def forward(self, src):
        out, hid = self.rnn(self.drop(self.emb(src)))
        hid = torch.tanh(self.fc(torch.cat((hid[-2], hid[-1]), dim=1)))
        return out, hid


class Decoder(nn.Module):
    def __init__(self, outdim, emb, eh, dh, drop, attention):
        super().__init__()
        self.outdim = outdim
        self.att = attention
        self.emb = nn.Embedding(outdim, emb)
        self.rnn = nn.GRU(eh * 2 + emb, dh)
        self.fc = nn.Linear(eh * 2 + dh + emb, outdim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, hidden, enc_out, mask):
        x = x.unsqueeze(0)
        emb = self.drop(self.emb(x))
        a = self.att(hidden, enc_out, mask).unsqueeze(1)
        enc = enc_out.permute(1, 0, 2)
        w = torch.bmm(a, enc).permute(1, 0, 2)
        out, hidden = self.rnn(torch.cat((emb, w), dim=2), hidden.unsqueeze(0))
        emb, out, w = emb.squeeze(0), out.squeeze(0), w.squeeze(0)
        return self.fc(torch.cat((out, w, emb), dim=1)), hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, spad, device):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.spad = spad
        self.device = device

    def mask(self, src):
        return (src != self.spad).permute(1, 0)

    def beam_search(self, src, sos_idx, eos_idx, beam_width=3, max_len=50):
        self.eval()
        with torch.no_grad():
            enc_out, hid = self.enc(src)
            m = self.mask(src)
            
            beams = [([sos_idx], 0.0, hid)]
            completed = []
            
            for _ in range(max_len):
                candidates = []
                for seq, score, hidden in beams:
                    if seq[-1] == eos_idx:
                        completed.append((seq, score))
                        continue
                    
                    x = torch.tensor([seq[-1]], device=self.device)
                    out, new_hidden, _ = self.dec(x, hidden, enc_out, m)
                    log_probs = torch.log_softmax(out, dim=1)
                    
                    topk_probs, topk_ids = torch.topk(log_probs[0], beam_width)
                    
                    for prob, idx in zip(topk_probs, topk_ids):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        candidates.append((new_seq, new_score, new_hidden))
                
                if not candidates:
                    break
                
                beams = heapq.nlargest(beam_width, candidates, key=lambda x: x[1])
                
                if all(seq[-1] == eos_idx for seq, _, _ in beams):
                    completed.extend((seq, score) for seq, score, _ in beams)
                    break
            
            if not completed:
                completed = [(seq, score) for seq, score, _ in beams]
            
            best_seq, best_score = max(completed, key=lambda x: x[1])
            return best_seq


def trim(tokens):
    out = []
    for t in tokens:
        if t == SOS:
            continue
        if t == EOS:
            break
        out.append(t)
    return out


def load_model_and_vocab(model_dir):
    model_dir = Path(model_dir)
    summary_path = model_dir / 'summary.json'
    model_path = model_dir / 'best_seq2seq_attention.pt'
    vocab_path = model_dir / 'vocab.json'

    if not summary_path.exists():
        raise FileNotFoundError(f"找不到 summary.json: {summary_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"找不到词表: {vocab_path}")

    with open(summary_path, encoding='utf-8') as f:
        summary = json.load(f)

    config = summary['config']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(vocab_path, encoding='utf-8') as f:
        vocab_data = json.load(f)
        src_vocab = Vocab(vocab_data['src_stoi'], vocab_data['src_itos'])
        trg_vocab = Vocab(vocab_data['trg_stoi'], vocab_data['trg_itos'])
        src_vocab.set_default_index(src_vocab[UNK])
        trg_vocab.set_default_index(trg_vocab[UNK])

    att = Attention(config['enc_hidden_dim'], config['dec_hidden_dim'])
    enc = Encoder(len(src_vocab.itos), config['emb_dim'], config['enc_hidden_dim'],
                  config['dec_hidden_dim'], config['dropout'])
    dec = Decoder(len(trg_vocab.itos), config['emb_dim'], config['enc_hidden_dim'],
                  config['dec_hidden_dim'], config['dropout'], att)
    model = Seq2Seq(enc, dec, src_vocab[PAD], device).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, src_vocab, trg_vocab, device, config


def translate_sentence(text, model, src_vocab, trg_vocab, device, max_len=50, beam_width=3):
    tokens = tokenize_zh(text)
    tokens = [SOS] + tokens + [EOS]
    src_ids = src_vocab.lookup_indices(tokens)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)

    pred_ids = model.beam_search(src_tensor, trg_vocab[SOS], trg_vocab[EOS], 
                                  beam_width=beam_width, max_len=max_len)
    pred_tokens = trim([trg_vocab.get_itos()[i] for i in pred_ids])

    return ' '.join(pred_tokens)


def interactive_mode(model_dir, beam_width=3):
    print("=" * 60)
    print("简易机器翻译系统 - 交互式翻译 (Beam Search)")
    print("=" * 60)
    print("\n正在加载模型...")

    model, src_vocab, trg_vocab, device, config = load_model_and_vocab(model_dir)

    print("[OK] 模型加载成功")
    print(f"[OK] 设备: {device}")
    print(f"[OK] 源语言词表大小: {len(src_vocab.itos)}")
    print(f"[OK] 目标语言词表大小: {len(trg_vocab.itos)}")
    print(f"[OK] Beam Width: {beam_width}")
    print("\n" + "=" * 60)
    print("使用说明:")
    print("  - 输入中文句子，系统会翻译成英文")
    print("  - 输入 'quit'、'exit' 或 'q' 退出")
    print("  - 输入 'example' 查看示例")
    print("=" * 60)

    examples = ["你好", "谢谢", "我爱你", "今天天气很好", "早上好", "我知道", "对不起"]

    while True:
        print()
        user_input = input("请输入中文句子 >>> ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n再见!")
            break

        if user_input.lower() == 'example':
            print("\n示例句子:")
            for ex in examples:
                print(f"  - {ex}")
            continue

        if not user_input:
            print("请输入有效的句子")
            continue

        try:
            translation = translate_sentence(user_input, model, src_vocab, trg_vocab,
                                            device, max_len=50, beam_width=beam_width)
            print(f"\n原文: {user_input}")
            print(f"译文: {translation}")
        except Exception as e:
            print(f"\n翻译出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='简易机器翻译系统 - 交互式翻译')
    parser.add_argument('--model-dir', type=str, default='outputs_zh_en',
                       help='模型目录路径')
    parser.add_argument('--beam-width', type=int, default=3,
                       help='Beam Search 宽度，默认3')
    args = parser.parse_args()

    interactive_mode(args.model_dir, args.beam_width)


if __name__ == '__main__':
    main()
