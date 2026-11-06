# 待处理问题清单(写报告前/提交前必须过一遍)

按**必须处理 → 建议处理 → 可选**排序。每条给了:问题是什么、证据在哪、怎么处理。

---

## A. 阻塞提交的问题(必须做,否则交不上)

### A1. ⚠️ 图已齐备;`data/` 仍缺失

- ✅ **8 张图已从 ipynb 中提取到 `metrial/fig/`,并已逐张核对过**:图上数值与 `03`/`05` 的表格完全一致。**写报告配图不再需要重跑 notebook。** 核对发现的三个排版小瑕疵(图例遮挡 `n=` 标注等)见 `06_figures.md` §「核对结果」。
- ⛔ **`data/` 仍然不存在。** 题目要求提交 *"your code and **data and figures** as a .ipynb file"*。
- **怎么处理:** 把 `data/` 下的 6 个 CSV(从 Kaggle 竞赛页面或队友处取得)、`figures/`(= 现在的 `fig/`,改名回 `figures/`)、带完整输出的 `.ipynb` 一起放进名为 **`midterm`** 的文件夹,提交到 **`midterm-code`**。
- **注意:不需要为了配图而重跑 notebook。** 现有的 notebook 输出是所有数字的唯一来源;**如果重跑且随机种子或库版本不同,数字可能变动**,反而要回头核对 `03`/`05`。既然图已经有了,**建议直接用现有输出,不要重跑**。

### A2. ⛔ 报告本身还没写;deadline 2026-11-07 23:59 ET

- 三个交付物:Kaggle ×2(已有提交文件)、报告 PDF → `midterm-report`、代码+数据+图 → `midterm-code`。
- **封面必须有:课程名、全部组员姓名、以及⭐每位组员负责的工作。** 后者容易漏,`00_report_requirements.md` 有原文。
- **必须使用题目指定的模板**;正文 **≤5 页含图**(References 可溢出到第 6 页)。

---

## B. 报告内容必须修正的事实性错误

### B1. ⭐⭐ notebook 的结论与数字矛盾:pooled 模型到底帮了哪些 phase

- **notebook markdown 原文:** "Phase-specific copies on top of shared weights help **phases I and II** the most"
- **实际数字**(`05_part2_results.md` §2(g)):
  | | per-phase field-wise | pooled | 变化 |
  |---|---|---|---|
  | AUC I | 0.7638 | 0.7585 | **−0.005(变差)** |
  | AUC II | 0.7246 | 0.7365 | +0.012 |
  | AUC III | 0.7758 | 0.7886 | **+0.013(最大)** |
- ⇒ **实际是 II 和 III 受益,I 略微变差。报告里必须写对。**
- **处理方式:** 不只是改掉,而是**把它变成一个 insight** —— phase I 数据最少却没受益,反而略降,说明增益来源不是"样本量补足"而是"相邻 phase 信号可迁移",而 phase I 的成功判据(安全性/剂量)与 II/III(疗效)差异最大。完整写法见 `07_discussion_insights.md` 论点 7。

### B2. ⭐ test 试验的年份范围写错了

- notebook 文字:test trials start in **2014–2021**
- 题目原文(`_raw_spec.txt`):**2013–2022**
- ⇒ **报告里用题目的 2013–2022。**(Part 1 的 test 年份 **2010–2016** 是对的,不要改。)

### B3. ⭐ Part 1 的模型选择存在 val / CV 分歧,不能简单说"H 最好"

- **数字**(`03_part1_results.md` §4):

  | | val s-F1 | val m-F1 | CV s-F1 | CV m-F1 |
  |---|---|---|---|---|
  | A supervised LR C=0.1 | 0.8590 | 0.7291 | **0.8957** | **0.7979** |
  | C K-means∩LR K=40 | 0.8718 | 0.7461 | **0.8967** | 0.7972 |
  | **H ensemble(已选定)** | **0.8974** | **0.8058** | 0.8900 | 0.7714 |

- notebook 写的 "its CV score is within noise of the best" **对 sample-F1 成立(差 0.0057),对 macro-F1 偏乐观(差 0.027)**。
- **处理方式(不要改模型,改表述):** 保留 H,理由三条 ——(1) 题目指定用 val 选模;(2) H 的优势集中在稀有类且有机制解释(Funding 0.2857→0.75、Other 全表最高 0.600);(3) CV 每折只有 ~188 个训练样本,伪标签过滤在小样本下更不稳,CV 低估了 H。**同时明确承认 78 个验证 trial 无法在统计上分辨 A / C / H。** 建议英文写法已备在 `07_discussion_insights.md` 论点 4。

---

## C. 必须主动声明的合规与设定问题(不声明容易被误解为违规)

### C1. ⭐⭐ 转导设定(transductive):无标签池包含 test 的特征

- **代码事实:** `XU = np.vstack([X_train[~labeled], X_test])` —— 半监督方法的无标签池**包含 test 的 `emb_*` 特征**,但**不含任何 test 标签**。
- **为什么要主动写:** 题目禁止 *"reverse-engineer the test labels"* 和 *"find the data source online and train on the fully labeled dataset"*。**我们做的都不是这两件事** —— 使用 test 的**特征**(不含标签)是标准的转导式半监督设定。但如果不说明,审阅者可能误解。
- **顺带要说明由此产生的不对称:** `XU` **不含** `X_val`,**也不含** CV 的留出折 ⇒ **val/CV 分数是归纳式(inductive)的,而 test 预测是转导式的** ⇒ 真实 test 表现可能略优于验证估计。**这是一个对我们有利的说明,写出来没有坏处。**

### C2. ⭐ Part 2 从未使用 reason 标签或 embedding 作为输入

- 题目硬约束:*"Reason embeddings and reason labels should be used only for error analysis, not as predictive inputs to the binary classifier."*
- notebook 2.3 节开头已写 "Reason labels and embeddings are never used as inputs here"。**报告里要有一句显式声明。**
- 同时声明:**`ID` / `trial_id` 只用于把失败试验匹配到 Part 1 的 reason 标签做错误分析,从未作为预测特征**(题目也点名了 "IDs only for matching, not as a predictive feature")。

### C3. 没有使用深度学习

题目:*"Do not use deep learning"*(若实现则只允许 PyTorch)。我们全程只用 sklearn 的线性模型 / NB / kNN / K-means / GMM / 图传播,**没有任何神经网络**。一句话声明即可。

---

## D. 需要在报告里说明、否则读者会误解的技术细节

### D1. ⭐ `fit_lr` 里 `class_weight="balanced"` 与 `sample_weight` 是**相乘**的

```python
def fit_lr(X, Y, C=0.1, w=None, class_weight="balanced"):
```
- sklearn 的 `LogisticRegression.fit(sample_weight=w)` 与 `class_weight` **是乘法关系**,不是二选一。
- ⇒ **伪标签的 `w_u = 0.3` 不是字面意义上的 0.3 权重**,它是 `0.3 × class_weight[y]`。对稀有类(Funding)的 `class_weight` 很大,所以**稀有类的伪标签实际权重远高于 0.3**。
- **处理方式:** 在 Methods 里写一句 "pseudo-label weights are applied on top of the balanced class weights, so the effective weight of a pseudo-label is 0.3 × the class weight"。**不要改代码**(现有工作不动),只是说清楚。这反而解释了为什么降权对稀有类伪标签的保护没那么强。

### D2. ⭐ ss-GMM 的分量数 K 没有在任何输出里打印出来

- 代码:`combos, cls = np.unique(YL, axis=0, return_inverse=True); K = len(combos)`
  即 **K = 有标签集合里出现过的不同 label-set 的个数**(不是预设的常数)。
- **问题:Methods 被明确要求写 "how many components / how they were initialized",但这个数字当前环境算不出来**(没有数据)。
- **怎么补:** 重跑 notebook 时加一行 `print(K)`,或者直接从 `part1_train_recovered_labels.csv` 的 `label_source == "true"` 的 157 行里数不同 label-set 的个数。**粗估上界:147 个单标签 trial 覆盖 ≤5 种,10 个双标签 trial 覆盖 ≤10 种 ⇒ K ≤ 15,实际大概 10–13。报告里必须填真实数字,不要写估计值。**
- **初始化方式本身已经清楚可写:** 不用随机初始化、也不用 k-means —— **M 步先只用有标签数据的 one-hot 责任矩阵算一次参数(`params = mstep(np.zeros((len(ZU), K)))`),以此作为 EM 的起点**。这是比随机初始化更强的"标签播种"(label-seeded)初始化。**收敛判据:相对对数似然变化 < 1e-5,上限 100 次,实际 70 次收敛。**

### D3. `p1_kmeans_diag.png` 没有画 ARI,而 ARI 是最有说服力的一列

- 图只画了 inertia / silhouette / purity;**ARI(K=5 峰值 0.2032 后单调下降)只在表里有**。
- **两个选项:**(a) 重跑时把 ARI 加成第四个子图;(b) 不改图,**在正文里用文字给出 ARI 的数字**。
- 推荐 (b),因为改图要重跑且占版面;ARI 一句话("ARI peaks at only 0.2032 for K=5 and declines for larger K")就能承载全部论证。

### D4. `p2_err_*.png` 里 phase III 的 Funding 组是空柱子(n=0)

- 表里是 **NaN**,图里是**空白柱子 + "n=0" 标注**。
- ⇒ **放这些图时必须在图注里写一句**,否则看起来像画图出错。

### D5. `make_pooled` 的缓存键用了 `id(train_df)`

```python
key = (id(train_df), C, cw, frozenset(fk.items()))
```
- `id()` 在对象被 GC 后可能被新对象复用 ⇒ 理论上会取到错误的缓存。
- **在当前代码里是安全的**(`train_df` 在整个循环期间都被引用,不会被回收)。
- ⇒ **不需要改(现有工作不动),也不需要写进报告。** 这里只是备个案,万一重跑出现诡异结果,这是第一个怀疑对象。

---

## E. 建议补做的实验(都很便宜,能直接提分)

> 这些都**不修改现有工作**,只是在 notebook 末尾追加 cell。如果时间紧,**全部跳过也能交**,但 E1 性价比很高。

### E1. ⭐ DBSCAN(题目建议清单里有,我们没做)

- 题目 Part 1 的建议算法清单:*"Gaussian Mixture Models, DBSCAN, K-Means"*。**我们做了 K-means 和 GMM,DBSCAN 缺席。**
- **代价极低**(几行代码),而且**几乎可以肯定会失败** —— §1.4 已经证明 silhouette 只有 0.16–0.31、没有密度明确的簇,DBSCAN 在这种空间里会把绝大多数点标成噪声(label = −1)。
- ⇒ **一个负面结果恰好佐证了 §1.4 的诊断,而且证明我们覆盖了题目建议的全部三种聚类算法。** 建议做,报告里一句话 + 一个噪声点比例的数字即可。

### E2. Part 1 的 KNN(题目建议清单里有,我们没做)

- 题目 Part 1 的分类器建议清单:*"Multinomial Naive Bayes, One-vs-Rest Logistic Regression, KNN"*。我们做了 OvR LR,**MNB 不适用(embedding 有负值,MNB 要求非负)**,KNN 缺席。
- **建议至少跑一个** `KNeighborsClassifier` 包在 `OneVsRestClassifier` 里(cosine 距离),补全覆盖面。
- **顺带说明 MNB 为什么没做:** `emb_*` 是 PCA + 随机旋转后的值,**有负数,MultinomialNB 不接受负特征**。这是一个有依据的排除理由,**写进报告比默默跳过好**。

### E3. Part 1 的多标签阈值

- 提交里多标签率 **16.0%(50/312)**,而训练集真实多标签率只有 **6.4%(10/157)**。
- **一个便宜的改进:** 第二个标签要求更高的阈值(或要求与最高概率的差距足够小)。
- 如果不做,**至少写进 future work**(`07_discussion_insights.md` 第六部分已备)。

### E4. Part 2 的 per-phase 分位数阈值

- phase III 在所有阈值下的 FPR 都显著高于 I/II(base rate 0.653 的后果)。
- 用 per-phase 分位数阈值而不是统一 0.5 可以均衡 FPR。**不影响 Kaggle 的 log loss**,所以纯属 insight 层面的补充。可只写进 future work。

### E5. field 权重没有做搜索

`{criteria: 1.0, diseases: 0.5, drugs: 0.5, icd: 0.5}` 是先验选择。**报告里如实说明"未调优"即可**,不必补做 —— `05_part2_results.md` §2(e) 已显示 field-wise 相对单一 TF-IDF 增益很小,说明权重不是关键因素。

---

## F. 报告写作层面的提醒

1. **5 页含图极度紧张。** 选图优先级见 `06_figures.md`:必放 `p1_kmeans_diag.png` + `p2_err_compare.png`,强烈建议 `p1_ssgmm.png`,**不要同时放 `p1_methods.png` 和 §4 的 9 行表**(Results 明确要求表格,选表)。
2. **Results 必须有汇总表。** Part 1 用 `03_part1_results.md` §4 的 9 行表;Part 2 用 `05_part2_results.md` §3 的 baseline-vs-improved 表。**30 行和 40 行的完整扫描表不要进正文。**
3. **Methods 的三条硬性要求不要漏:**(a) K-means 质心如何初始化(k-means++, `n_init=10`, `random_state=42`);(b) GMM 分量如何初始化(**标签播种:用有标签数据的 one-hot 责任做一次 M 步** —— 见 D2);(c) **EM 的收敛阈值与迭代次数(相对容差 1e-5,上限 100,实际 70 次)**。
4. **负面结果要写,不要藏。** 题目明确说 *"Even if the change does not improve the F1 score, discuss the results"*。我们有五个有价值的负面结果:ensemble 在 Part 2 无增益、shrinkage 最优 s=1.0、`class_weight="balanced"` 略差、三类领域知识特征全无效、λ=0 优于 λ=1。**这些是 Insight 分数的主要来源。**
5. **数字不要凭记忆写。** 当前环境无法重跑 notebook,`03`/`05` 两个文件里的数字是从 `_raw_notebook_dump.txt` 逐个抄出来的,是唯一可信来源。
6. **Discussion 里 Part 1 和 Part 2 有一条可以贯穿的主线:** 两部分都发现"描述试验**设计**的特征,对预测**结果/原因**几乎没有增量信息"。Part 1 是 criteria/disease/drug 加不进去,Part 2 是通路/靶点/指纹加不进去。**把这条写成一句统一的结论,比分开讲两次更有分量。**

---

## G. 快速检查清单(提交前逐项打勾)

- [ ] 报告 PDF,**≤5 页**(含图),用指定模板
- [ ] 封面:课程名 + **全部组员姓名** + **每人负责的工作**
- [ ] 五个章节齐全:Abstract / Methods / Results / Discussion / References
- [ ] Methods 写了 **K-means 质心初始化**、**GMM 分量初始化**、**EM 收敛阈值 + 迭代次数**
- [ ] Results 有**汇总表**(Part 1 和 Part 2 各一张)
- [ ] Discussion 写了**不同 reason 组的错误率差异** + **类别规模** + **预测 reason 标签可能有错**
- [ ] Discussion 写了 **val→test 差距**(时间划分实验的量化结果)
- [ ] 写了**负面结果**(至少 shrinkage、class weighting、领域知识特征)
- [ ] 显式声明:**Part 2 未使用 reason 标签/embedding 作为输入**;**ID 仅用于匹配**;**未使用深度学习**
- [ ] 说明**转导设定**(无标签池含 test 特征、不含 test 标签)
- [ ] 修正 **pooled 模型受益 phase(II/III,不是 I/II)**
- [ ] 修正 **test 年份 2013–2022**
- [ ] 坦白 **Part 1 的 val/CV 分歧**
- [ ] 图注里说明 **phase III Funding n=0 的空柱子**
- [x] 8 张图已提取到 `metrial/fig/` 并核对一致
- [ ] 配图图注里补上被图例挡住的 `n=` 数字(`p2_err_compare` / `p2_err_balanced` 的 Phase III)
- [ ] 补齐 `data/`(6 个 CSV)
- [ ] 打包名为 **`midterm`** 的文件夹(`.ipynb` + `data/` + `figures/`)→ `midterm-code`
- [ ] 两个 Kaggle 提交(`part1_submission.csv` 312 行、`part2_submission.csv` 1,500 行)
- [ ] 报告 PDF → `midterm-report`
