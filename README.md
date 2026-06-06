# 简易机器翻译系统：基于中英平行语料的 Seq2Seq Attention 实验

## 1. 项目目标

本项目围绕“实现一个机器翻译模型，并体验从数据准备、模型构建、训练调试到评估分析的全过程”这一要求展开。主任务为 **中文到英文机器翻译**，使用公开中英平行语料库 OPUS OpenSubtitles，构建并训练一个端到端的神经机器翻译系统。

本项目重点不在于追求很高的翻译分数，而在于通过实践观察模型行为，理解编码器-解码器架构和注意力机制在机器翻译中的作用。实验过程中不仅记录 BLEU 分数和损失曲线，也导出了注意力热力图、成功案例和失败案例，用于分析模型在哪些情况下表现较好，在哪些情况下容易出错。

本项目完成的核心内容包括：

- 选择公开中英平行语料库并进行文本清洗；
- 将原始中英文本整理为本地 CSV 数据集；
- 实现 `Encoder-Decoder` 结构的 `Seq2Seq` 模型；
- 在解码器中集成 `Bahdanau Attention` 注意力机制；
- 在训练集上训练模型，并通过验证集观察损失变化；
- 在测试集上计算 BLEU 分数；
- 可视化特定句子的注意力权重；
- 对成功案例和失败案例进行定量与定性分析。

---

## 2. 核心任务对应关系

| 项目要求 | 本项目实现 |
|---|---|
| 数据准备：选择公开中英平行语料库并清洗 | 使用 OPUS OpenSubtitles 中英平行语料，整理为 `zh,en` 两列 CSV |
| 模型实现：编码器-解码器或 Transformer | 实现基于 GRU 的 Seq2Seq 编码器-解码器模型 |
| 注意力机制 | 在解码器中实现 Bahdanau Attention |
| 模型训练与调试 | 在训练集训练，通过验证集监控损失，保存最优模型状态 |
| 损失曲线 | 输出 `outputs_zh_en/loss_curve.png` |
| BLEU 评价 | 使用 `sacrebleu` 在测试集计算 BLEU |
| 注意力可视化 | 输出成功/失败案例注意力热力图 |
| 案例分析 | 导出 `successful_cases.csv`、`failed_cases.csv`、`translation_analysis.csv` |
| 深度分析 | 分析短句成功、长句/专名/低频词/重复生成等失败模式 |

---

## 3. 项目结构

```text
project2_mt/
├─ train_mt.py                         # 主程序：数据处理、模型训练、测试、评估、可视化
├─ convert_opus_to_csv.py              # 将 OPUS 原始中英文本转换为 CSV
├─ requirements.txt                    # Python 依赖
├─ README.md                           # 项目说明
├─ SUBMISSION_GUIDE.md                 # 文件说明与提交指南
├─ data/
│  └─ zh_en_opensubtitles.csv          # 整理后的中英 CSV 数据
├─ en-zh_CN.txt/                       # OPUS 原始语料目录
│  ├─ LICENSE
│  ├─ README
│  ├─ OpenSubtitles.en-zh_CN.en
│  └─ OpenSubtitles.en-zh_CN.zh_CN
└─ outputs_zh_en/                      # 中英主实验结果
   ├─ best_seq2seq_attention.pt
   ├─ summary.json
   ├─ loss_curve.png
   ├─ attention_heatmap_best_case.png
   ├─ attention_heatmap_failed_case.png
   ├─ translation_analysis.csv
   ├─ successful_cases.csv
   └─ failed_cases.csv
```


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

完整依赖见：

```text
requirements.txt
```

### 4.3 安装依赖

```bash
conda activate nlp_hmq
pip install -r requirements.txt
```

本项目当前不依赖 `torchtext`。原因是 Windows 环境下 `torchtext` 容易出现动态库兼容问题，因此项目使用自定义词表类完成词表构建和 token-id 转换。

---

## 5. 数据准备

### 5.1 数据集来源

本项目使用 OPUS 平台提供的 OpenSubtitles 中英平行语料。

- 语料库：OpenSubtitles
- 来源平台：OPUS
- 语言对：English - Chinese Simplified / `en-zh_CN`
- 任务方向：中文 → 英文

下载并解压后，原始文件包括：

```text
en-zh_CN.txt/OpenSubtitles.en-zh_CN.zh_CN
en-zh_CN.txt/OpenSubtitles.en-zh_CN.en
```

两个文件按行对齐：第 1 行中文对应第 1 行英文，第 2 行中文对应第 2 行英文，以此类推。

### 5.2 转换为 CSV

项目使用 `convert_opus_to_csv.py` 将 OPUS 原始语料转换为本地 CSV：

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

训练脚本会进行如下处理：

1. 读取 `zh` 和 `en` 两列；
2. 删除中文或英文为空的样本；
3. 去除首尾空白字符；
4. 打乱数据顺序；
5. 划分训练集、验证集和测试集；
6. 中文采用字符级切分；
7. 英文使用正则表达式切分单词、数字和标点；
8. 添加 `<sos>` 和 `<eos>` 序列边界标记；
9. 构建源语言词表和目标语言词表；
10. 将 token 序列转换为 id，并在 batch 内 padding。

本次最终实验使用的数据规模为：

| 数据集 | 样本数 |
|---|---:|
| 训练集 | 8000 |
| 验证集 | 1000 |
| 测试集 | 1000 |

---

## 6. 模型方法

### 6.1 整体架构

本项目采用 `Seq2Seq + Bahdanau Attention` 模型。模型由三部分组成：

1. **Encoder**  
   使用双向 GRU 编码中文源句，得到每个输入位置的上下文表示。

2. **Attention**  
   在解码每个英文 token 时，根据当前解码器隐状态计算源句各位置的注意力权重。

3. **Decoder**  
   使用 GRU 逐步生成英文译文。每一步输入包括当前 token、上一时刻隐状态和注意力上下文向量。

### 6.2 为什么使用注意力机制

传统 Seq2Seq 模型往往将源句压缩为一个固定长度向量。对于长句或信息密集的句子，这种压缩容易造成信息丢失。注意力机制允许解码器在生成每一个目标 token 时重新查看源句不同位置，从而缓解长句信息瓶颈问题。

同时，注意力权重可以被可视化为热力图，用来观察模型在生成目标词时关注了源句中的哪些位置。这对理解模型行为、分析成功和失败案例很有帮助。

---

## 7. 训练与调试

### 7.1 运行命令

最终中英实验运行命令如下：

```bash
python "f:\NLP\project2_mt\train_mt.py" --local-csv "f:\NLP\project2_mt\data\zh_en_opensubtitles.csv" --src-col zh --tgt-col en --epochs 3 --batch-size 64 --max-train-samples 8000 --max-valid-samples 1000 --max-test-samples 1000 --output-dir "f:\NLP\project2_mt\outputs_zh_en"
```

如果在项目目录内运行，可写为：

```bash
python train_mt.py --local-csv "data/zh_en_opensubtitles.csv" --src-col zh --tgt-col en --epochs 3 --batch-size 64 --max-train-samples 8000 --max-valid-samples 1000 --max-test-samples 1000 --output-dir "outputs_zh_en"
```

### 7.2 关键超参数

| 参数 | 设置 |
|---|---:|
| Epoch | 3 |
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
outputs_zh_en/best_seq2seq_attention.pt
```

训练完成后，脚本会加载最优模型，在测试集上生成译文并计算 BLEU。

---

## 8. 核心实验结果

### 8.1 损失变化

| Epoch | Train Loss | Valid Loss | Train PPL | Valid PPL | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0260 | 4.6403 | 152.3193 | 103.5730 | 17m34s |
| 2 | 4.4271 | 4.4812 | 83.6843 | 88.3450 | 15m41s |
| 3 | 4.1393 | 4.3965 | 62.7607 | 81.1687 | 8m23s |

训练损失和验证损失均持续下降，说明模型在训练过程中确实学习到了部分中英句子对应关系。验证损失没有出现明显反弹，因此在当前 3 轮训练内没有观察到严重过拟合。

损失曲线保存于：

```text
outputs_zh_en/loss_curve.png
```

### 8.2 BLEU 分数

测试集 BLEU 结果如下：

| 指标 | 数值 |
|---|---:|
| BLEU | 2.7763 |
| 1-gram precision | 21.0039 |
| 2-gram precision | 4.8091 |
| 3-gram precision | 1.7164 |
| 4-gram precision | 0.3427 |
| BP | 1.0000 |
| sys_len | 12312 |
| ref_len | 9081 |

BLEU 分数较低，但这与实验设置和数据特点是一致的。本项目使用 CPU 训练 3 轮，模型为基础 RNN Seq2Seq + Attention，中文采用字符级切分，且 OpenSubtitles 字幕语料存在口语化、省略、上下文依赖和对齐噪声。因此，本实验重点是完整实现机器翻译流程，并通过案例分析理解模型能力边界，而不是追求工业级翻译质量。

---

## 9. 注意力可视化

项目会对成功案例和失败案例分别生成注意力热力图：

```text
outputs_zh_en/attention_heatmap_best_case.png
outputs_zh_en/attention_heatmap_failed_case.png
```

注意力热力图的横轴为中文源句 token，纵轴为模型生成的英文 token。颜色越深，表示模型在生成该英文 token 时越关注对应的中文输入位置。

在成功案例中，注意力通常更集中，模型能够较好关注到关键源语言位置。失败案例中，注意力可能分散或集中在无关位置，常伴随 `<unk>`、重复生成或过长输出。

---

## 10. 案例分析

### 10.1 成功案例

成功案例保存于：

```text
outputs_zh_en/successful_cases.csv
```

部分结果如下：

| 中文源句 | 英文参考 | 模型预测 | Token F1 |
|---|---|---|---:|
| 谁？ | Who? | who ? | 1.0000 |
| 是的。 Yes. | Yes. | yes . | 1.0000 |
| 我没哭 | I'm not crying. | i ' m not . | 0.9091 |
| 我知道。 | I know. | i know know . | 0.8571 |
| 等一會兒. | Wait a minute. | a minute . | 0.8571 |

这些例子说明，模型对短句、高频词和常见口语表达有一定学习能力。例如“谁？”能够被翻译为“who ?”，说明模型已经学习到部分直接的中英词语对应关系。

### 10.2 失败案例

失败案例保存于：

```text
outputs_zh_en/failed_cases.csv
```

典型失败案例如下：

| 中文源句 | 英文参考 | 主要问题 |
|---|---|---|
| 音乐 戈特福里德·休帕茨 雕塑 沃尔特·舒尔茨 | Music: Gottfried Huppertz. Sculptor: | 出现大量 `<unk>` |
| 真的 你指望我相信你吗 | Really? Expect me to believe that? | 输出大量重复 `you` |
| 党同伐异 | Intolerance | 专有片名无法正确翻译 |

失败模式主要包括：

1. **长句处理较弱**  
   长句需要模型保持更多上下文信息，基础 RNN 解码器容易丢失远距离信息。

2. **专有名词和低频词较难翻译**  
   人名、地名、片名等低频词在训练集中出现次数少，容易被映射为 `<unk>`。

3. **字幕语料噪声影响较大**  
   OpenSubtitles 中存在省略句、口语句、上下文依赖句和对齐不完全严格的句子。

4. **重复生成问题**  
   贪心解码和基础 RNN 解码器有时会反复输出高频 token，如 `you you you`。

5. **字符级中文切分粒度较细**  
   字符级切分虽然简单稳定，但缺少词级语义边界，对成语、专名和固定搭配不够友好。

这些现象符合基础 Seq2Seq 模型的典型局限，也说明通过失败案例能够观察模型能力边界。

---

## 11. 输出成果

本项目最终输出包括：

### 11.1 方法简述

- 数据来源：OPUS OpenSubtitles 中英平行语料；
- 数据格式：`zh,en` 两列 CSV；
- 模型结构：Seq2Seq + Bahdanau Attention；
- 编码器：双向 GRU；
- 解码器：GRU；
- 评价方式：BLEU、损失曲线、注意力图、案例分析。

### 11.2 核心结果

```text
outputs_zh_en/summary.json
outputs_zh_en/loss_curve.png
outputs_zh_en/attention_heatmap_best_case.png
outputs_zh_en/attention_heatmap_failed_case.png
```

### 11.3 深度分析文件

```text
outputs_zh_en/translation_analysis.csv
outputs_zh_en/successful_cases.csv
outputs_zh_en/failed_cases.csv
```

---

## 12. 如何复现实验

### 12.1 准备数据

如果已经有整理好的 CSV：

```text
data/zh_en_opensubtitles.csv
```

可直接训练。

如果只有 OPUS 原始文件，则先运行：

```bash
python convert_opus_to_csv.py
```

### 12.2 训练中英模型

```bash
python train_mt.py --local-csv "data/zh_en_opensubtitles.csv" --src-col zh --tgt-col en --epochs 3 --batch-size 64 --max-train-samples 8000 --max-valid-samples 1000 --max-test-samples 1000 --output-dir "outputs_zh_en"
```

### 12.3 查看结果

训练完成后查看：

```text
outputs_zh_en/summary.json
outputs_zh_en/loss_curve.png
outputs_zh_en/successful_cases.csv
outputs_zh_en/failed_cases.csv
outputs_zh_en/attention_heatmap_best_case.png
outputs_zh_en/attention_heatmap_failed_case.png
```

---

## 13. 项目结论

本项目完成了一个端到端的中英机器翻译实验。从公开中英平行语料的获取和清洗开始，到模型构建、训练调试、测试评估和案例分析，整个流程均已实现。

实验结果表明，`Seq2Seq + Bahdanau Attention` 模型可以学习到部分短句和高频表达的中英对应关系。训练损失和验证损失持续下降，说明模型训练过程有效；注意力热力图和案例分析进一步展示了模型在具体句子上的行为。与此同时，模型在长句、专有名词、低频词和字幕噪声较大的句子上容易失败，这反映出基础编码器-解码器模型的典型局限。

总体而言，本项目达到了课程项目要求：模型能正常训练并收敛，能计算 BLEU，能生成注意力图，并能对成功与失败案例进行有依据的分析。

---

## 14. 后续改进方向

后续可以从以下方向改进：

- 增加训练数据规模；
- 增加训练轮数；
- 使用 GPU 加速训练；
- 引入 `jieba`、BPE 或 SentencePiece；
- 使用 Beam Search 改进解码；
- 使用 Transformer 替代 RNN Seq2Seq；
- 对 OpenSubtitles 语料进行更严格的清洗；
- 针对长句、否定结构、代词指代和成语设计更多测试案例。

---

## 15. 附加实验：Multi30k 英德翻译模式

除中英主任务外，脚本仍保留 Multi30k 英德翻译模式，用于验证模型流程的通用性。

如果不传入 `--local-csv`，脚本默认尝试使用 `bentrevett/multi30k` 数据集进行英文到德文翻译：

```bash
python train_mt.py --epochs 5 --batch-size 128 --max-train-samples 12000 --max-valid-samples 1000 --max-test-samples 1000
```

早期英德示例结果保存在：

```text
outputs/
```

最终以 `outputs_zh_en/` 中的中英实验结果为主，英德实验只作为额外说明。

---

## 16. 参考资料

1. OPUS OpenSubtitles 语料库：  
   https://opus.nlpl.eu/

2. Jörg Tiedemann. Parallel Data, Tools and Interfaces in OPUS. LREC, 2012.

3. Pierre Lison, Jörg Tiedemann. OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. LREC, 2016.

4. Ilya Sutskever, Oriol Vinyals, Quoc V. Le. Sequence to Sequence Learning with Neural Networks. NIPS, 2014.

5. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR, 2015.

6. Papineni K., Roukos S., Ward T., Zhu W. BLEU: a Method for Automatic Evaluation of Machine Translation. ACL, 2002.
