# 报告硬性要求(逐条摘自 Midterm_Project_2026.pdf)

来源:`/home/dev/compal/Midterm_Project_2026.pdf`(8 页),原文已提取到 `_raw_spec.txt`。

## 截止与提交物

- **截止时间:2026 年 11 月 7 日(周六)23:59 ET**,提交到 Gradescope(从 Canvas 进)和 Kaggle。
- 三项提交物:
  1. Kaggle 上提交 Part 1 和 Part 2 的预测(两个独立的竞赛)。
  2. 报告,**单个 .pdf 文件**,提交到 Gradescope 的 `midterm-report` 作业。
  3. 源码与数据文件(**以及图**),`.ipynb` 格式,放在名为 **`midterm`** 的文件夹里,提交到 `midterm-code` 作业。

> 注意第 3 项要求把 **数据文件和图一起交**。8 张图已在 `metrial/fig/`;`data/` 下的 6 个 CSV 打包前需要补回来。

## 报告格式

- **不超过 5 页,含图**。参考文献不计入,可以溢出到第 6 页且数量不限。
- 低于页数上限是可以的,原文明确说 "We appreciate conciseness"。
- 必须使用指定模板(PDF 里给了链接),目的是统一字号等版式,便于评分。
- **封面页**必须包含:课程名、全体组员姓名、以及**每位组员负责的工作**。

## 必须有的五个章节(原文给的顺序和要求)

| 章节 | 原文要求 |
|---|---|
| **1. Abstract** | 简述项目做了什么以及观察到的结果。 |
| **2. Methods** | 写清模型细节。明确点名两类内容:1) 模型本身,例如**聚类质心或 GMM 分量是如何初始化的**;2) 训练过程,例如 **EM 算法判定收敛的阈值参数、迭代次数**等。 |
| **3. Results** | 实验结果,**必须包含一张汇总各模型性能的表**。 |
| **4. Discussion** | 项目中获得的洞察,例如什么带来了好的性能,以及为什么会这样的假设。 |
| **5. References** | 列出所有用到的论文、书籍、网站(思路、代码、措辞都算)。遵守 Canvas 上的学术诚信政策。 |

另有通用要求:包含与问题相关的全部内容(文字、公式、图、图表、输出);**如果放了图,要附上生成该图的源码**。

## 评分细则(总分 100,各项之和可超过 100,最终封顶 100)

| 项 | 分值 | 评分点 |
|---|---|---|
| **Creativity** | 35 | 方法是否有创意/多样?是否实现并尝试了多种算法?是否比较了多个**算法类别**?原文给的"算得上有创意"的例子是:下面每一类都实现并比较**至少两个**选项 —— 2 种 reason-labeling 方法、2 种二分类器或设置、2 种特征/预处理选择。也可以做 **ensemble**。 |
| **Insight** | 35 | 是否揭示了超参数、架构等对性能的影响?对结果是否有合理的解释/直觉?(原文:不能只是盲目地把不同算法套上去然后比较。)**Part 2 要讨论不同 reason 类别间错误率的差异,并考虑类别规模和预测 reason 标签本身可能有错。** |
| **Performance** | 30 | 在可用算力下性能是否相对不错?不同算法如何比较?**是否对选定的算法做了优化**,而不是随便训几个模型就完事?**Part 2 要评估改进措施及其 trade-off。** |
| **Report Writeup** | 15 | 方法、洞察、结果是否清晰呈现与解释。 |

## 建议的算法清单(原文列出,用于对照 Creativity)

无监督(至少要评估其中之一):
- (a) Distribution based clustering,例如 **GMM**
- (b) Density based clustering,例如 **DBSCAN**(课上没讲,但可以试)
- (c) Centroid-based clustering,例如 **K-Means**

监督(可组合使用):
- (a) **Multinomial Naive Bayes**(Part 2,配非负文本特征)
- (b) **One-vs-rest Logistic Regression**(Part 1)或 binary LR(Part 2)
- (c) **K Nearest Neighbors**(Part 1 多标签;Part 2 二分类)

Part 2 的文本特征表示:
- (a) Bag of words  (b) N-gram  (c) TF-IDF

## 禁止事项

- **不得使用深度学习方法**(原文:为了对没上过 ML 课的同学公平,期末项目才可以用)。
- 如需自己实现算法,只能用 **PyTorch**,不得用 Keras/TensorFlow。(本项目没有涉及。)
- **不得试图反推 test set 的标签。**
- **不得去网上找数据源、用全标签数据集训练。**
- 违反上述两条会被严重扣分。
- 允许的库:scikit-learn、scikit-image、numpy、scipy、**rdkit**(用于 SMILES 列)。
- 任何对公开代码仓库的引用或复制都必须正确引用(GitHub、Wikipedia、博客等)。

## Part 1 的任务要求原文要点

- 用**至少一种无监督方法**(可选择与监督方法结合)来自动为缺失标签的样本补标签(即增广数据集)。
- 为 `part1_train.csv` 生成缺失的 reason 标签。
- 用 `part1_val.csv` 的表现来**选模型和特征组合**。
- 预测 `part1_test.csv` 的五个 reason 标签。
- **讨论选了哪些特征,以及为什么某些特征没能提升性能 / 为什么"全都加上"可能反而不提升。**(这一点在 PDF 里出现了两次,是明确的得分点。)
- 评估指标:**sample-wise F1**(按 trial 平均),并报告**每个 reason 标签的 F1** 和 **macro-averaged F1**。
- Kaggle 提交格式:每个 trial 一行,列为 `ID` 和 `Predicted`,`Predicted` 用空格分隔多个 reason(例:`Recruitment Other`)。

## Part 2 的任务要求原文要点

- 为每个 trial phase(I/II/III)构建二分类器预测试验结果。
- **"Success 在不同 phase 含义不同,所以要么每个 phase 训一个单独的模型,要么训一个使用 phase 的模型。"**(原文允许两种做法 —— 这正是 pooled phase-augmented LR 的合法性依据,报告里值得引用这句。)
- 用 Part 1 的 reason 标签分析预测错误。
- 改进分类器。**即使改动没有提升 F1 也要讨论结果。**
- 报告每个 phase 的 **验证集 F1、ROC-AUC、PR-AUC**(average precision),以 label=1 为正类。F1 由二值预测算,ROC-AUC 和 PR-AUC 由 label=1 的预测概率算。
- Kaggle 提交:`outcome_test.csv` 每行 label=1 的预测概率,列为 `ID` 和 `label`。**Kaggle 分数是 log loss**,因为每个 phase 的 test 行数相同,所以它等于各 phase log loss 的平均。
- **原文明确要求讨论 val→test 的性能差距**("Expect lower performance on the test set than on the validation set, since the test trials are from later years; discuss this gap in your report")。
- 可以考虑用 `start_year` 作为额外预测特征。

### 错误分析的具体规定

- 对 `outcome_val.csv` 中 label=0 的失败试验,用 `trial_id` 匹配 Part 1 的 `ID`,取其已有或预测的 reason 标签。
- 很多失败试验是跑完了但没达到终点、并未提前停止,因此不在 Part 1 中;**把它们作为单独的 "no recorded stopping reason" 组**。
- 一个 trial 可以有多个 reason,所以可以同时计入多个类别。
- **每个 phase 分别分析**,使得一个出现在两个 phase 的 trial 在单个 phase 内不被重复计数。
- **必须说明哪些 reason 标签是真标签、哪些是你 Part 1 的预测。**
- 对每个 reason 类别(含 no-reason 组)和每个 phase,报告 **trial 数量** 和 **被错误预测为成功(label=1)的比例**。
- 基于上述分析改进分类器,并在 **F1、ROC-AUC、PR-AUC** 以及各 reason 类别的错误率上与原模型对比。

### 硬约束

> **"Reason embeddings and reason labels should be used only for error analysis, not as predictive inputs to the binary classifier."**

报告里最好明确写一句"我们遵守了这一约束",因为这是容易被怀疑的点。

## ID 使用约束

- Trial 用匿名 ID(如 `TR00042`),NCT 号未提供且已从文本字段中移除。
- **同一个 trial 在两部分中 ID 相同**:Part 1 的 `ID` 列对应 Part 2 的 `trial_id` 列。
- **这些 ID 只能用于识别和匹配 trial,不能作为预测特征。**
</content>
</invoke>
