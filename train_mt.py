import argparse, json, math, os, random, time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns, spacy, torch
import sacrebleu
import torch.nn as nn, torch.optim as optim
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

HF_MIRROR = 'https://hf-mirror.com'
UNK, PAD, SOS, EOS = '<unk>', '<pad>', '<sos>', '<eos>'


def set_seed(seed=1234):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def tok(ex, en_nlp, de_nlp, max_length):
    en = [t.text.lower() for t in en_nlp.tokenizer(ex['en'])][:max_length]
    de = [t.text.lower() for t in de_nlp.tokenizer(ex['de'])][:max_length]
    return {'src_tokens': [SOS, *en, EOS], 'trg_tokens': [SOS, *de, EOS]}


def y(ds, key):
    for item in ds: yield item[key]


def y(ds, key):
    for item in ds: yield item[key]


class Vocab:
    def __init__(self, stoi, itos, default_index=0):
        self.stoi = stoi
        self.itos = itos
        self.default_index = default_index

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi[token]

    def set_default_index(self, idx):
        self.default_index = idx

    def lookup_indices(self, tokens):
        return [self.stoi.get(token, self.default_index) for token in tokens]

    def get_itos(self):
        return self.itos


def build_vocab_from_iterator(iterator, min_freq=1, specials=None):
    specials = specials or []
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    itos = []
    seen = set()
    for token in specials:
        if token not in seen:
            itos.append(token)
            seen.add(token)
    for token, freq in counter.items():
        if freq >= min_freq and token not in seen:
            itos.append(token)
            seen.add(token)
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi, itos)


def num(ds, sv, tv):
    out = []
    for item in ds:
        out.append({'en': item['en'], 'de': item['de'], 'src_tokens': item['src_tokens'], 'trg_tokens': item['trg_tokens'],
                    'src_ids': sv.lookup_indices(item['src_tokens']), 'trg_ids': tv.lookup_indices(item['trg_tokens'])})
    return out


def collate(spad, tpad):
    def f(batch):
        src = pad_sequence([torch.tensor(x['src_ids']) for x in batch], padding_value=spad)
        trg = pad_sequence([torch.tensor(x['trg_ids']) for x in batch], padding_value=tpad)
        return src.long(), trg.long()
    return f


def trim(tokens):
    out = []
    for t in tokens:
        if t == SOS: continue
        if t == EOS: break
        out.append(t)
    return out


def f1(ref, pred):
    rc, pc = Counter(ref), Counter(pred)
    ov = sum((rc & pc).values())
    if ov == 0: return 0.0
    p, r = ov / max(len(pred), 1), ov / max(len(ref), 1)
    return 2 * p * r / (p + r)


class Attention(nn.Module):
    def __init__(self, eh, dh):
        super().__init__(); self.attn = nn.Linear(eh * 2 + dh, dh); self.v = nn.Linear(dh, 1, bias=False)
    def forward(self, hidden, enc_out, mask):
        sl = enc_out.shape[0]; hidden = hidden.unsqueeze(1).repeat(1, sl, 1); enc = enc_out.permute(1, 0, 2)
        e = torch.tanh(self.attn(torch.cat((hidden, enc), dim=2))); a = self.v(e).squeeze(2)
        return torch.softmax(a.masked_fill(mask == 0, -1e10), dim=1)


class Encoder(nn.Module):
    def __init__(self, inp, emb, eh, dh, drop):
        super().__init__(); self.emb = nn.Embedding(inp, emb); self.rnn = nn.GRU(emb, eh, bidirectional=True); self.fc = nn.Linear(eh * 2, dh); self.drop = nn.Dropout(drop)
    def forward(self, src):
        out, hid = self.rnn(self.drop(self.emb(src))); hid = torch.tanh(self.fc(torch.cat((hid[-2], hid[-1]), dim=1))); return out, hid


class Decoder(nn.Module):
    def __init__(self, outdim, emb, eh, dh, drop, attention):
        super().__init__(); self.outdim = outdim; self.att = attention; self.emb = nn.Embedding(outdim, emb); self.rnn = nn.GRU(eh * 2 + emb, dh); self.fc = nn.Linear(eh * 2 + dh + emb, outdim); self.drop = nn.Dropout(drop)
    def forward(self, x, hidden, enc_out, mask):
        x = x.unsqueeze(0); emb = self.drop(self.emb(x)); a = self.att(hidden, enc_out, mask).unsqueeze(1); enc = enc_out.permute(1, 0, 2)
        w = torch.bmm(a, enc).permute(1, 0, 2); out, hidden = self.rnn(torch.cat((emb, w), dim=2), hidden.unsqueeze(0))
        emb, out, w = emb.squeeze(0), out.squeeze(0), w.squeeze(0)
        return self.fc(torch.cat((out, w, emb), dim=1)), hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, spad, device):
        super().__init__(); self.enc = enc; self.dec = dec; self.spad = spad; self.device = device
    def mask(self, src): return (src != self.spad).permute(1, 0)
    def forward(self, src, trg, tfr=0.5):
        b, tl, vs = src.shape[1], trg.shape[0], self.dec.outdim; outs = torch.zeros(tl, b, vs, device=self.device)
        enc_out, hid = self.enc(src); x = trg[0]; m = self.mask(src)
        for t in range(1, tl):
            out, hid, _ = self.dec(x, hid, enc_out, m); outs[t] = out; x = trg[t] if random.random() < tfr else out.argmax(1)
        return outs
    def translate(self, src, sos_idx, eos_idx, max_len=50):
        self.eval(); attn = []
        with torch.no_grad():
            enc_out, hid = self.enc(src); m = self.mask(src); x = torch.tensor([sos_idx], device=self.device); toks = [sos_idx]
            for _ in range(max_len):
                out, hid, a = self.dec(x, hid, enc_out, m); p = out.argmax(1).item(); toks.append(p); attn.append(a.squeeze(0).cpu().numpy())
                if p == eos_idx: break
                x = torch.tensor([p], device=self.device)
        return toks, np.stack(attn, axis=0) if attn else np.zeros((0, src.shape[0]))

def train_epoch(model, loader, opt, crit, clip, device):
    model.train(); total = 0.0
    for src, trg in tqdm(loader, desc='train', leave=False):
        src, trg = src.to(device), trg.to(device)
        opt.zero_grad(); out = model(src, trg); dim = out.shape[-1]
        loss = crit(out[1:].view(-1, dim), trg[1:].reshape(-1)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step(); total += loss.item()
    return total / max(len(loader), 1)


def eval_epoch(model, loader, crit, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for src, trg in tqdm(loader, desc='valid', leave=False):
            src, trg = src.to(device), trg.to(device)
            out = model(src, trg, 0.0); dim = out.shape[-1]
            total += crit(out[1:].view(-1, dim), trg[1:].reshape(-1)).item()
    return total / max(len(loader), 1)


def ids2tokens(ids, vocab):
    itos = vocab.get_itos()
    return [itos[i] for i in ids]


def translate_set(model, rows, tv, device, max_len):
    sos_idx, eos_idx = tv[SOS], tv[EOS]
    preds, refs, details = [], [], []
    for item in tqdm(rows, desc='test', leave=False):
        src = torch.tensor(item['src_ids'], dtype=torch.long, device=device).unsqueeze(1)
        pred_ids, att = model.translate(src, sos_idx, eos_idx, max_len=max_len)
        ptok, rtok = trim(ids2tokens(pred_ids, tv)), trim(item['trg_tokens'])
        preds.append(' '.join(ptok)); refs.append([' '.join(rtok)])
        details.append({'source_en': item['en'], 'reference_de': item['de'], 'predicted_de': ' '.join(ptok), 'reference_tokens': rtok, 'predicted_tokens': ptok, 'attention': att.tolist(), 'source_tokens': item['src_tokens']})
    return preds, refs, details


def save_loss_curve(hist, path):
    df = pd.DataFrame(hist)
    plt.figure(figsize=(8, 5)); plt.plot(df['epoch'], df['train_loss'], marker='o', label='train'); plt.plot(df['epoch'], df['valid_loss'], marker='o', label='valid')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def save_attention(row, path):
    att = np.array(row['attention']); pred = row['predicted_tokens']; src = trim(row['source_tokens'])
    if att.size == 0 or not pred:
        return
    xlab = [SOS, *src, EOS]; att = att[:len(pred), :len(xlab)]
    plt.figure(figsize=(max(8, len(xlab) * 0.6), max(4, len(pred) * 0.5)))
    sns.heatmap(att, cmap='YlGnBu', xticklabels=xlab, yticklabels=pred)
    plt.xlabel('Source tokens'); plt.ylabel('Predicted target tokens'); plt.title('Attention Weights'); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--enc-hidden-dim', type=int, default=256)
    ap.add_argument('--dec-hidden-dim', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--max-length', type=int, default=50)
    ap.add_argument('--min-freq', type=int, default=2)
    ap.add_argument('--max-train-samples', type=int, default=12000)
    ap.add_argument('--max-valid-samples', type=int, default=1000)
    ap.add_argument('--max-test-samples', type=int, default=1000)
    args = ap.parse_args()

    os.environ.setdefault('HF_ENDPOINT', HF_MIRROR); set_seed(args.seed)
    base = Path(__file__).resolve().parent; outdir = base / 'outputs'; outdir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Using device: {device}')
    en_nlp, de_nlp = spacy.load('en_core_web_sm'), spacy.load('de_core_news_sm')
    ds = load_dataset('bentrevett/multi30k')
    train, valid, test = ds['train'], ds['validation'], ds['test']
    if args.max_train_samples: train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_valid_samples: valid = valid.select(range(min(args.max_valid_samples, len(valid))))
    if args.max_test_samples: test = test.select(range(min(args.max_test_samples, len(test))))
    kw = {'en_nlp': en_nlp, 'de_nlp': de_nlp, 'max_length': args.max_length}
    train, valid, test = train.map(tok, fn_kwargs=kw), valid.map(tok, fn_kwargs=kw), test.map(tok, fn_kwargs=kw)

    specials = [UNK, PAD, SOS, EOS]
    sv = build_vocab_from_iterator(y(train, 'src_tokens'), min_freq=args.min_freq, specials=specials)
    tv = build_vocab_from_iterator(y(train, 'trg_tokens'), min_freq=args.min_freq, specials=specials)
    sv.set_default_index(sv[UNK]); tv.set_default_index(tv[UNK])
    train_rows, valid_rows, test_rows = num(train, sv, tv), num(valid, sv, tv), num(test, sv, tv)
    loader = collate(sv[PAD], tv[PAD])
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=loader)
    valid_loader = DataLoader(valid_rows, batch_size=args.batch_size, shuffle=False, collate_fn=loader)

    att = Attention(args.enc_hidden_dim, args.dec_hidden_dim)
    enc = Encoder(len(sv), args.emb_dim, args.enc_hidden_dim, args.dec_hidden_dim, args.dropout)
    dec = Decoder(len(tv), args.emb_dim, args.enc_hidden_dim, args.dec_hidden_dim, args.dropout, att)
    model = Seq2Seq(enc, dec, sv[PAD], device).to(device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    crit = nn.CrossEntropyLoss(ignore_index=tv[PAD])

    hist, best = [], float('inf')
    best_path = outdir / 'best_seq2seq_attention.pt'
    for ep in range(1, args.epochs + 1):
        st = time.time(); tr = train_epoch(model, train_loader, opt, crit, args.clip, device); va = eval_epoch(model, valid_loader, crit, device)
        mins, secs = divmod(int(time.time() - st), 60)
        hist.append({'epoch': ep, 'train_loss': tr, 'valid_loss': va, 'train_ppl': math.exp(tr), 'valid_ppl': math.exp(va), 'minutes': mins, 'seconds': secs})
        print(f'Epoch {ep:02d} | Time {mins}m {secs}s | Train Loss {tr:.3f} | Val Loss {va:.3f}')
        if va < best:
            best = va; torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    preds, refs, details = translate_set(model, test_rows, tv, device, args.max_length)
    bleu_result = sacrebleu.corpus_bleu(preds, list(map(list, zip(*refs))))
    bleu = {'bleu': bleu_result.score, 'precisions': bleu_result.precisions, 'bp': bleu_result.bp, 'sys_len': bleu_result.sys_len, 'ref_len': bleu_result.ref_len}

    rows = []
    for row in details:
        score = f1(row['reference_tokens'], row['predicted_tokens'])
        rows.append({'source_en': row['source_en'], 'reference_de': row['reference_de'], 'predicted_de': row['predicted_de'], 'reference_len': len(row['reference_tokens']), 'prediction_len': len(row['predicted_tokens']), 'token_f1': round(score, 4), 'length_gap': abs(len(row['reference_tokens']) - len(row['predicted_tokens']))})
    df = pd.DataFrame(rows).sort_values(['token_f1', 'length_gap'], ascending=[False, True])
    ok = df.head(10).copy(); bad = df.sort_values(['token_f1', 'length_gap'], ascending=[True, False]).head(10).copy()
    df.to_csv(outdir / 'translation_analysis.csv', index=False, encoding='utf-8-sig')
    ok.to_csv(outdir / 'successful_cases.csv', index=False, encoding='utf-8-sig')
    bad.to_csv(outdir / 'failed_cases.csv', index=False, encoding='utf-8-sig')
    save_loss_curve(hist, outdir / 'loss_curve.png')
    if details:
        save_attention(next(x for x in details if x['source_en'] == ok.iloc[0]['source_en']), outdir / 'attention_heatmap_best_case.png')
        save_attention(next(x for x in details if x['source_en'] == bad.iloc[0]['source_en']), outdir / 'attention_heatmap_failed_case.png')

    summary = {'device': str(device), 'config': vars(args), 'vocab_sizes': {'src': len(sv), 'trg': len(tv)}, 'dataset_sizes': {'train': len(train_rows), 'valid': len(valid_rows), 'test': len(test_rows)}, 'best_valid_loss': best, 'bleu': bleu, 'history': hist}
    with open(outdir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'BLEU: {bleu["bleu"]:.4f}')
    print(f'Outputs saved to: {outdir}')


if __name__ == '__main__':
    main()

