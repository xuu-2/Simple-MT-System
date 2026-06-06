from pathlib import Path
import csv

base = Path(r'f:\NLP\project2_mt')
zh_path = base / 'en-zh_CN.txt' / 'OpenSubtitles.en-zh_CN.zh_CN'
en_path = base / 'en-zh_CN.txt' / 'OpenSubtitles.en-zh_CN.en'
out_path = base / 'data' / 'zh_en_opensubtitles.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)

limit = 14000
kept = 0
seen = set()
with open(zh_path, 'r', encoding='utf-8', errors='ignore') as zh_f, open(en_path, 'r', encoding='utf-8', errors='ignore') as en_f, open(out_path, 'w', encoding='utf-8-sig', newline='') as out_f:
    writer = csv.DictWriter(out_f, fieldnames=['zh', 'en'])
    writer.writeheader()
    for zh, en in zip(zh_f, en_f):
        zh = zh.strip()
        en = en.strip()
        if not zh or not en:
            continue
        if len(zh) < 2 or len(en) < 2 or len(zh) > 80 or len(en) > 140:
            continue
        pair = (zh, en.lower())
        if pair in seen:
            continue
        seen.add(pair)
        writer.writerow({'zh': zh, 'en': en})
        kept += 1
        if kept >= limit:
            break
print(f'Saved {kept} sentence pairs to {out_path}')
