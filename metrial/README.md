# 报告素材库索引

CS5785 / ORIE5750 / ECE5414 Fall 2026 Midterm Project 报告写作材料。
**用途:** 这里面是从题目 PDF 和 notebook 已保存输出中提取、整理、解读后的全部素材,供手写报告时直接取用。
**原则:** 本目录只新增文件,**没有修改任何现有工作**(`CS5785_Midterm.ipynb`、`midterm_src.py`、三个提交 CSV 均未动过)。

---

## 文件清单

| 文件 | 内容 | 对应报告章节 |
|---|---|---|
| **`00_report_requirements.md`** | 题目对报告的全部要求:deadline、三个交付物、页数限制、模板、封面要求、五个章节的具体规定、评分细则(Creativity 35 / Insight 35 / Performance 30 / Writeup 15)、建议算法清单、**禁止事项** | 全局 |
| **`01_dataset_facts.md`** | 全部数据事实,逐条标注 **[题目]**(来自 PDF)或 **[实测]**(来自 notebook 输出):样本量、划分、标签可见性机制、embedding 来源、各列语义、prevalence、base rate、匹配率 | Methods 的数据描述 |
| **`02_part1_methods.md`** | Part 1 的九个方法家族(A–H)的完整配置与代码,**⭐ 标出了评分细则点名要求的项**(质心初始化、GMM 初始化、EM 收敛阈值与迭代次数)、评估协议、特征实验设计 | Methods(Part 1) |
| **`03_part1_results.md`** | Part 1 的全部数字:baseline、11 行聚类诊断表、**40 行完整方法扫描表**、**9 行家族最优汇总表**、EM 收敛、6 行特征表、最终模型逐标签对比、prevalence 漂移检查、提交分布 | Results(Part 1) |
| **`04_part2_methods.md`** | Part 2 的方法细节:合规声明、三种文本表示、三种分类器、**field-wise TF-IDF**、**ICD chapter-prefix 技巧**、RDKit Morgan 指纹、**pooled phase-augmented LR**、错误分析方法(逐条对应题目要求)、时间稳健性实验设计 | Methods(Part 2) |
| **`05_part2_results.md`** | Part 2 的全部数字:baseline 表、**30 行完整模型表**、**baseline vs improved 汇总表**、**两张 21 行错误分析表**、阈值 trade-off 表、class weighting 结果、**时间漂移实验**、提交分布 | Results(Part 2) |
| **`06_figures.md`** | 8 张图的清单、生成 cell、技术细节,以及 **5 页限制下的选图优先级建议**和版面预算 | 配图决策 |
| **`07_discussion_insights.md`** | **九条可直接成段的核心论点**(含假设与解释)、Part 1 特征选择论证、Part 2 错误分析的六点解读、时间漂移的三条结论、局限与 future work | **Discussion** |
| **`08_references.md`** | 18 条参考文献(含 BibTeX)+ **按版面优先级的取舍建议**(最小集 5 条,推荐集 10 条) | References |
| **`09_issues_to_address.md`** | **⚠️ 写报告前务必先读。** 阻塞提交的问题、**必须修正的两处事实性错误**、必须主动声明的合规说明、容易被误解的技术细节、建议补做的廉价实验、**提交前检查清单** | 全局 |

### 原始材料(未加工,备查)

| 文件 | 内容 |
|---|---|
| `_raw_spec.txt` | 题目 PDF 的逐字文本(8 页) |
| `_raw_notebook_dump.txt` | notebook 的全部 cell + 全部输出(1,482 行)。**所有数字的唯一可信来源** |
| `fig/` | 从 ipynb 提取出的 8 张结果图(PNG),文件名与 notebook 的 `FIG = "figures/"` 一致 |

---

## 建议的阅读/写作顺序

1. **`09_issues_to_address.md`** —— 先看有哪些必须修正和必须声明的东西,避免写完再改。
2. **`00_report_requirements.md`** —— 确认章节结构、页数、封面、评分权重。
3. 写 Methods:**`01` + `02` + `04`**。注意 `02`/`04` 里的 ⭐ 标记是硬性评分要求。
4. 写 Results:**`03` §4 + `05` §3** 两张汇总表进正文,配图按 **`06`** 的优先级选。
5. 写 Discussion:**`07`** 直接按论点取用。
6. 写 References:**`08`** 按优先级表取舍。
7. 提交前走 **`09` §G 的检查清单**。

---

## 两个重要前提

1. **当前环境无法重跑 notebook**(`data/` 缺失,且无 pandas / scikit-learn)。因此 `03` 和 `05` 里的数字全部逐个抄自 `_raw_notebook_dump.txt`,**写报告时不要凭记忆改动小数位**。既然图已齐备,**不建议重跑** —— 重跑若种子/版本不同,数字反而要重新核对。
2. **8 张图已在 `fig/`**(已逐张核对,数值与表格一致);**`data/` 下的 6 个 CSV 仍需在打包 `midterm-code` 前补齐**。详见 `09_issues_to_address.md` §A1。
