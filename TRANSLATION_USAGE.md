# 交互式中英翻译系统使用指南

## 概述

这个交互式翻译系统允许你输入中文句子，实时得到英文翻译。

## 模型效果评价

根据训练结果：

- **BLEU分数**: 2.78
- **训练损失**: 从 5.03 降到 4.14
- **验证损失**: 从 4.64 降到 4.40
- **能力特点**:
  - ✓ 能翻译简单短句，如"谁？" → "who?"
  - ✓ 能识别高频词汇
  - ✗ 长句效果差
  - ✗ 专有名词会变成 `<unk>`
  - ✗ 容易重复生成词语

## 使用前准备

1. 确保已经训练好模型，并且存在以下文件：
   - `outputs_zh_en/best_seq2seq_attention.pt`（模型权重）
   - `outputs_zh_en/summary.json`（配置信息）
   - `outputs_zh_en/vocab.json`（词表文件）

2. 如果没有 `vocab.json`，运行：

```bash
python generate_vocab.py --model-dir outputs_zh_en
```

## 交互式翻译

启动交互式翻译界面：

```bash
conda activate nlp_hmq
python interactive_translate.py --model-dir outputs_zh_en
```

然后输入中文句子测试：

```
请输入中文句子 >>> 你好
原文：你好
译文：you .

请输入中文句子 >>> 谢谢
原文：谢谢
译文：thank you .

请输入中文句子 >>> 我爱你
原文：我爱你
译文：i love you .
```

输入 `q`、`quit` 或 `exit` 退出。

输入 `example` 查看示例句子。

## 批量翻译

将要翻译的句子写入文本文件（每行一句），然后：

```bash
python interactive_translate.py --model-dir outputs_zh_en --batch --input input.txt --output output.txt
```

## 建议测试的句子类型

### 适合的句子（预期效果较好）
- 短句：你好、谢谢、是的、不
- 简单表达：我知道、我爱你、你在哪里
- 常见口语：等一会儿、没关系

### 困难的句子（预期效果差）
- 长句：今天早上我去了超市买了很多东西然后回家做饭
- 专有名词：张三、北京、清华大学
- 成语：画蛇添足、守株待兔
- 复杂结构：如果明天不下雨的话我就去公园

## 模型局限性说明

1. **训练不充分**: 只训练了 3 轮，使用 CPU，未充分收敛
2. **数据噪声**: 字幕语料存在口语化、省略、对齐不准确
3. **分词粒度**: 中文采用字符级切分，语义粒度过细
4. **模型能力**: 基础 RNN Seq2Seq，不如 Transformer
5. **解码策略**: 贪心解码容易重复，未使用 Beam Search
6. **词表覆盖**: 低频词和专有名词会被映射为 `<unk>`

## 改进建议

如果想提高翻译质量：

1. 增加训练轮数（至少 10+ 轮）
2. 使用 GPU 加速训练
3. 增加训练数据量
4. 使用 jieba 或 BPE 改进中文分词
5. 采用 Beam Search 解码
6. 升级为 Transformer 架构
7. 使用预训练模型

## 文件说明

- `train_mt.py`: 训练脚本
- `interactive_translate.py`: 交互式翻译脚本
- `generate_vocab.py`: 词表生成脚本
- `outputs_zh_en/`: 中英翻译模型输出目录
  - `best_seq2seq_attention.pt`: 模型权重
  - `vocab.json`: 词表
  - `summary.json`: 实验配置和结果
  - `loss_curve.png`: 损失曲线图
  - `attention_heatmap_*.png`: 注意力图
  - `*_cases.csv`: 成功/失败案例

## 总结

这个模型是一个**教学实验性质的简易翻译系统**，主要用于：
- 理解 Seq2Seq + Attention 机制
- 观察模型训练过程
- 分析成功和失败案例
- 学习机器翻译评价方法

**不适合**用于实际生产环境的翻译任务。
