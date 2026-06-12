# 简易机器翻译系统：基于中英平行语料的 Seq2Seq Attention 实验

## 1. 项目目标

本项目围绕“实现一个机器翻译模型，并体验从数据准备、模型构建、训练调试到评估分析的全过程”这一要求展开。主任务为 **中文到英文机器翻译**，使用公开中英平行语料库 OPUS OpenSubtitles，构建并训练一个端到端的神经机器翻译系统。

本项目重点不在于追求工业级翻译效果，而在于通过实践观察模型行为，理解编码器-解码器架构和注意力机制在机器翻译中的作用。实验过程中记录 BLEU 分数和损失曲线，导出注意力热力图、成功案例和失败案例，并提供交互式翻译脚本用于手动测试。

本项目完成的核心内容包括：

- 选择公开中英平行语料库并进行文本清洗；
- 将原始中英文本整理为本地 CSV 数据集；
- 实现 `Encoder-Decoder` 结构的 `Seq2Seq` 模型；
- 在解码器中集成 `Bahdanau Attention` 注意力机制；
- 在训练集上训练模型，并通过验证集观察损失变化；
- 在测试集上计算 BLEU 分数；
- 可视化特定句子的注意力权重；
- 对成功案例和失败案例进行定量与定性分析；
- 增加 `interactive_translate.py`，支持用户输入中文并得到英文翻译；
- 增加 `analyze_attention.py`，针对否定结构、代词、长句和 `<unk>` 问题生成注意力分析。

---

## 2. 核心任务对应关系

| 项目要求 | 本项目实现 |
|---|---|
| 数据准备：选择公开中英平行语料库并清洗 | 使用 OPUS OpenSubtitles 中英平行语料，整理为 `zh,en` 两列 CSV |
| 模型实现：编码器-解码器或 Transformer | 实现基于 GRU 的 Seq2Seq 编码器-解码器模型 |
| 注意力机制 | 在解码器中实现 Bahdanau Attention |
| 模型训练与调试 | 训练集训练，通过验证集监控损失，保存最优模型状态 |
| 损失曲线 | 输出 `outputs_zh_en_5ep/loss_curve.png` |
| BLEU 评价 | 使用 `sacrebleu` 在测试集计算 BLEU，最终 BLEU 为 4.4473 |
| 注意力可视化 | 输出基础注意力图，并额外生成 6 类典型案例注意力热力图 |
| 案例分析 | 导出 `successful_cases.csv`、`failed_cases.csv`、`translation_analysis.csv` 和 `attention_analysis/case_analysis.csv` |
| 交互式测试 | 提供 `interactive_translate.py`，支持用户输入中文并翻译为英文 |

---

## 3. 项目结构

```text
project2_mt/
├─ train_mt.py                         # 主程序：训练、测试、评估、可视化
├─ convert_opus_to_csv.py              # OPUS 原始中英文本转换为 CSV
├─ generate_vocab.py                   # 根据训练数据重建 vocab.json
├─ interactive_translate.py            # 交互式翻译脚本，支持 Beam Search
├─ analyze_attention.py                # 注意力可视化和典型案例分析脚本
├─ requirements.txt                    # Python 依赖
├─ README.md                           # 项目说明
├─ GITHUB_GUIDE.md                     # GitHub 提交指南
├─ TRANSLATION_USAGE.md                # 翻译系统使用说明
├─ .gitignore                          # Git 忽略文件
├─ data/
│  └─ zh_en_opensubtitles.csv          # 整理后的中英 CSV 数据，上传 GitHub 时可忽略
├─ en-zh_CN.txt/                       # OPUS 原始语料目录，上传 GitHub 时可忽略
├─ outputs_zh_en_5ep/                  # 最终中英实验结果，5 轮训练
│  ├─ best_seq2seq_attention.pt        # 最优模型权重
│  ├─ vocab.json                       # 词表文件
│  ├─ summary.json                     # 实验配置、损失和 BLEU 结果
│  ├─ loss_curve.png                   # 损失曲线
│  ├─ attention_heatmap_best_case.png  # 自动成功案例注意力图
│  ├─ attention_heatmap_failed_case.png # 自动失败案例注意力图
│  ├─ translation_analysis.csv         # 测试集翻译分析
│  ├─ successful_cases.csv             # 成功案例
│  └─ failed_cases.csv                 # 失败案例
└─ attention_analysis/                 # 针对性注意力分析结果
   ├─ 1_success_short.png
   ├─ 2_success_polite.png
   ├─ 3_fail_negation.png
   ├─ 4_fail_pronoun.png
   ├─ 5_fail_long.png
   ├─ 6_fail_unk.png
   └─ case_analysis.csv
```

---

## 4. 实验环境

使用 `conda` 虚拟环境 `nlp_hmq` 运行本项目。

### 4.1 Python 版本

- Python 3.9

### 4.2 主要依赖

- `torch==2.1.0`
- `datasets==2.16.1`
- `spacy==3.7.4`
- `matplotlib==3.8.4`
- `seaborn==0.13.2`
- `pandas==2.2.2`
- `numpy==1.26.4`
- `tqdm==4.66.4`
- `sacrebleu==2.4.2`

完整依赖见 `requirements.txt`。

### 4.3 安装依赖

```bash
conda activate nlp_hmq
pip install -r requirements.txt
```

本项目当前不依赖 `torchtext`，而是使用自定义词表类完成词表构建和 token-id 转换。

---

## 5. 数据准备

### 5.1 数据集来源

本项目使用 OPUS 平台提供的 OpenSubtitles 中英平行语料。

- 语料库：OpenSubtitles
- 来源平台：OPUS
- 语言对：English - Chinese Simplified / `en-zh_CN`
- 任务方向：中文 → 英文

原始文件包括：

```text
en-zh_CN.txt/OpenSubtitles.en-zh_CN.zh_CN
en-zh_CN.txt/OpenSubtitles.en-zh_CN.en
```

两个文件按行对齐：同一行的中文句子和英文句子构成一组平行句对。

### 5.2 转换为 CSV

```bash
python convert_opus_to_csv.py
```

转换后的文件为：

```text
data/zh_en_opensubtitles.csv
```

CSV 格式如下：

```csv
zh,en
中文句子,English sentence
```

### 5.3 文本清洗和预处理

训练脚本会进行如下处理：读取 `zh` 和 `en` 两列、删除空样本、去除首尾空白、使用固定随机种子打乱数据、划分训练集/验证集/测试集、中文字符级切分、英文正则切词、添加 `<sos>` 和 `<eos>`、构建词表、转换为 id 并在 batch 内 padding。

本次最终实验使用的数据规模为：

| 数据集 | 样本数 |
|---|---:|
| 训练集 | 8000 |
| 验证集 | 1000 |
| 测试集 | 1000 |

---

## 6. 模型方法

### 6.1 整体架构

本项目采用 `Seq2Seq + Bahdanau Attention` 模型，由三部分组成：

1. **Encoder**：使用双向 GRU 编码中文源句，得到每个输入位置的上下文表示。
2. **Attention**：在解码每个英文 token 时，根据当前解码器隐状态计算源句各位置的注意力权重。
3. **Decoder**：使用 GRU 逐步生成英文译文，每一步结合当前 token、上一时刻隐状态和注意力上下文向量。

### 6.2 为什么使用注意力机制

传统 Seq2Seq 模型会将源句压缩为一个固定长度向量，长句中容易丢失信息。注意力机制允许解码器在生成每个目标 token 时重新查看源句不同位置，从而缓解固定向量瓶颈问题。

同时，注意力权重可以被可视化为热力图。即使模型翻译很差，也可以通过注意力图观察模型是否学到了合理对齐，或者说明注意力分布混乱导致翻译失败。

---

## 7. 训练与调试

### 7.1 运行命令

最终中英实验采用 5 轮训练：

```bash
python train_mt.py --local-csv "data/zh_en_opensubtitles.csv" --src-col zh --tgt-col en --epochs 5 --batch-size 64 --max-train-samples 8000 --max-valid-samples 1000 --max-test-samples 1000 --output-dir "outputs_zh_en_5ep" --seed 1234
```

训练完成后生成词表：

```bash
python generate_vocab.py --model-dir "outputs_zh_en_5ep" --data-csv "data/zh_en_opensubtitles.csv" --max-train-samples 8000 --seed 1234 --min-freq 2
```

### 7.2 关键超参数

| 参数 | 设置 |
|---|---:|
| Epoch | 5 |
| Batch Size | 64 |
| Embedding Dimension | 128 |
| Encoder Hidden Dimension | 256 |
| Decoder Hidden Dimension | 256 |
| Dropout | 0.5 |
| Learning Rate | 0.001 |
| Gradient Clip | 1.0 |
| Max Length | 50 |
| Min Frequency | 2 |
| 源语言词表大小 | 2234 |
| 目标语言词表大小 | 2548 |

### 7.3 模型训练状态

每轮训练结束后，脚本会在验证集上计算损失，并保存验证损失最低的模型参数：

```text
outputs_zh_en_5ep/best_seq2seq_attention.pt
```

实验中也尝试了 10 轮训练，但第 5 轮之后验证损失开始上升，说明模型出现过拟合，因此最终采用 5 轮模型。

---

## 8. 核心实验结果

### 8.1 损失变化

| Epoch | Train Loss | Valid Loss | Train PPL | Valid PPL | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.026 | 4.640 | 152.32 | 103.57 | 3m10s |
| 2 | 4.427 | 4.481 | 83.68 | 88.35 | 3m11s |
| 3 | 4.139 | 4.397 | 62.76 | 81.17 | 3m11s |
| 4 | 3.884 | 4.329 | 48.58 | 75.75 | 3m11s |
| 5 | 3.672 | 4.290 | 39.34 | 73.13 | 3m13s |

训练损失和验证损失在前 5 轮持续下降。第 5 轮验证损失最低，因此作为最终模型。10 轮实验中训练损失继续下降，但验证损失上升，说明继续训练会过拟合。

损失曲线保存于：

```text
outputs_zh_en_5ep/loss_curve.png
```

### 8.2 BLEU 分数

测试集 BLEU 结果如下：

| 指标 | 数值 |
|---|---:|
| BLEU | 4.4473 |
| 1-gram precision | 约 23% |
| 2-gram precision | 约 6% |
| 3-gram precision | 约 2% |
| 4-gram precision | 低于 1% |

BLEU 分数仍然较低，但比 3 轮训练时的 2.7763 有提升。这与实验设置和数据特点一致：本项目使用 CPU 训练，模型为基础 RNN Seq2Seq + Attention，中文采用字符级切分，且 OpenSubtitles 字幕语料存在口语化、省略、上下文依赖和对齐噪声。

---

## 9. 注意力可视化

### 9.1 基础注意力图

训练脚本会对成功案例和失败案例分别生成注意力热力图：

```text
outputs_zh_en_5ep/attention_heatmap_best_case.png
outputs_zh_en_5ep/attention_heatmap_failed_case.png
```

注意力热力图横轴为中文源句 token，纵轴为模型生成的英文 token。颜色越深，表示生成该英文 token 时越关注对应的中文输入位置。

### 9.2 针对性注意力分析

项目增加了 `analyze_attention.py`，专门生成 6 类典型案例的注意力图：

```bash
python analyze_attention.py --model-dir "outputs_zh_en_5ep" --output-dir "attention_analysis"
```

| 类型 | 输入示例 | 分析目的 |
|---|---|---|
| 成功案例 1 | 你好 | 观察短句是否形成清晰对齐 |
| 成功案例 2 | 谢谢 | 观察高频礼貌用语是否翻译正确 |
| 失败案例 1：否定结构 | 我不喜欢咖啡 | 检查生成时是否关注“不” |
| 失败案例 2：代词 | 她看见她的妈妈 | 检查 she/her 是否混淆 |
| 失败案例 3：长句 | 今天早上我去了超市买了很多东西然后回家做饭吃 | 观察是否漏译或重复 |
| 失败案例 4：词表外词 | 机器翻译 | 分析 `<unk>` 和词表覆盖不足问题 |

分析结果保存于：

```text
attention_analysis/
attention_analysis/case_analysis.csv
```

---

## 10. 案例分析

### 10.1 成功案例

成功案例保存于：

```text
outputs_zh_en_5ep/successful_cases.csv
```

部分结果如下：

| 中文源句 | 英文参考 | 模型预测 | Token F1 |
|---|---|---|---:|
| - 是的 冯·伯恩伯格夫人 | -Yes, Miss von Bernburg. | - yes , miss von bernburg . | 1.0000 |
| 我很抱歉。 我下去把它们扔出去。 | I'm sorry. | i ' m sorry . | 1.0000 |
| 快前进 | Forward! | forward ! | 1.0000 |
| 快看！ | Look! | look ! | 1.0000 |
| 你好? | Hello? | hello ? | 1.0000 |
| - 晚安. | - Good night. | - good night . | 1.0000 |
| 但我不用. | But I don't. | but i don ' t . . | 0.9231 |
| 我没哭 | I'm not crying. | i ' m not . | 0.9091 |

这些例子说明，模型对短句、高频词和常见口语表达有一定学习能力。例如“你好?”能够被翻译为“hello ?”，“快看！”能够被翻译为“look !”。

### 10.2 失败案例

失败案例保存于：

```text
outputs_zh_en_5ep/failed_cases.csv
```

典型失败案例如下：

| 中文源句 | 英文参考 | 模型预测 | 主要问题 |
|---|---|---|---|
| “明天，这花蕾将盛放” | This bud will blossom...tomorrow. | 连续重复引号 | 重复生成严重 |
| 这钱并不干净 Tony这些钱不是靠正档工作赚来的。 | Cesca! | that ' s not that ' s not `<unk>` . | 语料对齐噪声 + 重复 |
| 这些根本不够我们吃 | - Kat! Are you crazy? | this is this wouldn ' t we we ' t we ' re . | 句意错误，重复生成 |
| 下更多的蛋 下更多的蛋 | Lay more eggs! Lay more eggs! | the `<unk>` `<unk>` `<unk>` . | 低频词或语义未学习 |
| 隆重宣传 伊曼努尔·拉斯教授 | We'll hang large posters... | the `<unk>` `<unk>` `<unk>` . | 专有名词无法翻译 |

失败模式主要包括长句处理较弱、专有名词和低频词难翻译、字幕语料噪声较大、重复生成、字符级中文切分粒度较细等。

---

## 11. 交互式翻译测试

项目提供 `interactive_translate.py`，用户可以直接输入中文句子并得到模型翻译结果。脚本支持 Beam Search，能够一定程度上减少重复生成问题。

```bash
python interactive_translate.py --model-dir "outputs_zh_en_5ep" --beam-width 3
```

示例结果：

| 输入 | 模型输出 | 说明 |
|---|---|---|
| 你好 | hello , you . | 能识别 hello，但有多余词 |
| 谢谢 | thank you . | 翻译较好 |
| 早上好 | morning morning . | 出现重复 |
| 机器翻译 | the `<unk>` ' ll be `<unk>` . | 词表覆盖不足 |
| 今天天气很好 | well , good evening . . . | 语义偏离 |

---

## 12. 输出成果

### 12.1 方法简述

- 数据来源：OPUS OpenSubtitles 中英平行语料；
- 数据格式：`zh,en` 两列 CSV；
- 模型结构：Seq2Seq + Bahdanau Attention；
- 编码器：双向 GRU；
- 解码器：GRU；
- 评价方式：BLEU、损失曲线、注意力图、案例分析、交互式翻译。

### 12.2 核心结果

```text
outputs_zh_en_5ep/summary.json
outputs_zh_en_5ep/vocab.json
outputs_zh_en_5ep/loss_curve.png
outputs_zh_en_5ep/attention_heatmap_best_case.png
outputs_zh_en_5ep/attention_heatmap_failed_case.png
```

### 12.3 深度分析文件

```text
outputs_zh_en_5ep/translation_analysis.csv
outputs_zh_en_5ep/successful_cases.csv
outputs_zh_en_5ep/failed_cases.csv
attention_analysis/case_analysis.csv
attention_analysis/*.png
```

---

## 13. 如何复现实验

```bash
python convert_opus_to_csv.py
python train_mt.py --local-csv "data/zh_en_opensubtitles.csv" --src-col zh --tgt-col en --epochs 5 --batch-size 64 --max-train-samples 8000 --max-valid-samples 1000 --max-test-samples 1000 --output-dir "outputs_zh_en_5ep" --seed 1234
python generate_vocab.py --model-dir "outputs_zh_en_5ep" --data-csv "data/zh_en_opensubtitles.csv" --max-train-samples 8000 --seed 1234 --min-freq 2
python interactive_translate.py --model-dir "outputs_zh_en_5ep" --beam-width 3
python analyze_attention.py --model-dir "outputs_zh_en_5ep" --output-dir "attention_analysis"
```

---

## 14. 项目结论

本项目完成了一个端到端的中英机器翻译实验。从公开中英平行语料的获取和清洗开始，到模型构建、训练调试、测试评估、交互式翻译和注意力案例分析，整个流程均已实现。

实验结果表明，`Seq2Seq + Bahdanau Attention` 模型可以学习到部分短句和高频表达的中英对应关系。5 轮训练后，测试集 BLEU 达到 4.4473，相比 3 轮训练有所提升。注意力热力图和案例分析进一步展示了模型在具体句子上的行为。

与此同时，模型在长句、否定结构、代词区分、专有名词、低频词和字幕噪声较大的句子上容易失败。这反映出基础编码器-解码器模型的典型局限，也说明本项目更适合作为机器翻译流程和注意力机制的教学实验，而不是实际生产级翻译系统。

---


## 15. 后续改进方向

后续可以从以下方向改进：增加训练数据规模，使用 GPU 加速训练，引入 `jieba`、BPE 或 SentencePiece 改进中文分词，使用 Beam Search 改进所有测试阶段解码，使用 Transformer 替代 RNN Seq2Seq，对 OpenSubtitles 语料进行更严格清洗，或使用预训练机器翻译模型进行微调。

---

## 16. 附加实验：Multi30k 英德翻译模式

除中英主任务外，脚本仍保留 Multi30k 英德翻译模式，用于验证模型流程的通用性。如果不传入 `--local-csv`，脚本默认尝试使用 `bentrevett/multi30k` 数据集进行英文到德文翻译。

最终以 `outputs_zh_en_5ep/` 中的中英实验结果为主，英德实验只作为额外说明。

---

## 17. 参考资料

1. OPUS OpenSubtitles 语料库：https://opus.nlpl.eu/
2. Jörg Tiedemann. Parallel Data, Tools and Interfaces in OPUS. LREC, 2012.
3. Pierre Lison, Jörg Tiedemann. OpenSubtitles2016. LREC, 2016.
4. Ilya Sutskever, Oriol Vinyals, Quoc V. Le. Sequence to Sequence Learning with Neural Networks. NIPS, 2014.
5. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR, 2015.
6. Papineni K., Roukos S., Ward T., Zhu W. BLEU: a Method for Automatic Evaluation of Machine Translation. ACL, 2002.
