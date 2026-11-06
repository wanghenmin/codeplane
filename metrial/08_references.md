# References(参考文献素材)

**题目规定:References 不计入 5 页正文限制,可以溢出到第 6 页。** 所以不必为了省版面而删引用。
格式题目没有强制指定,用一种一致的格式即可(下面按 ACM/IEEE 风格编号写,也给了 BibTeX)。

---

## 1. 必引(数据来源与题目直接涉及)

**[1] 数据集来源 —— HINT**
Fu, T., Huang, K., Xiao, C., Glass, L. M., & Sun, J. (2022). *HINT: A hierarchical interaction network for clinical-trial outcome prediction.* **Patterns**, 3(4), 100445.

> 这是 Part 2 的 outcome 标签以及 drug-knowledge 列(targets / actions / pathways / SMILES)的来源。题目明确提到 "we corrected 7 labels that contradicted the trial's ClinicalTrials.gov record, but some label noise likely remains" —— **引用它时要一并提到标签噪声,这是解释性能上限的依据。**

```bibtex
@article{fu2022hint,
  title   = {{HINT}: A hierarchical interaction network for clinical-trial outcome prediction},
  author  = {Fu, Tianfan and Huang, Kexin and Xiao, Cao and Glass, Lucas M. and Sun, Jimeng},
  journal = {Patterns},
  volume  = {3},
  number  = {4},
  pages   = {100445},
  year    = {2022},
  doi     = {10.1016/j.patter.2022.100445}
}
```

**[2] 试验元数据来源**
ClinicalTrials.gov, U.S. National Library of Medicine. https://clinicaltrials.gov

> `criteria` / `phase` / `start_year` / 停止原因文本 / 结局判定的原始来源。

**[3] ICD 编码体系**
World Health Organization. *International Statistical Classification of Diseases and Related Health Problems (ICD-10)*. https://icd.who.int

> 引它是为了支撑 **ICD chapter-prefix 技巧**(`E11.65 → "E11 E11.65"`):层级编码里前三位是章节/类目,所以前缀共享是有依据的,不是随意的字符串切割。

**[4] Reactome 通路数据库**
Gillespie, M., Jassal, B., Stephan, R., et al. (2022). *The Reactome pathway knowledgebase 2022.* **Nucleic Acids Research**, 50(D1), D687–D692.

> `drug_pathways` 列的 `R-HSA-*` ID 和 `pathway_level` 序数深度的来源。**引它时最好一并说明 `pathway_level` 描述的是通路本身的层级深度,不是药物属性** —— 这是我们只把它用于筛 level-0 顶层通路、而不当连续特征的理由。

**[5] DrugBank**
Wishart, D. S., Feunang, Y. D., Guo, A. C., et al. (2018). *DrugBank 5.0: a major update to the DrugBank database for 2018.* **Nucleic Acids Research**, 46(D1), D1074–D1082.

> `drug_targets` / `drug_actions` / `smiless` 的典型来源(经 HINT 整理)。如果不确定 HINT 具体用了哪个版本,可以在正文里只说 "drug-knowledge fields distributed with the dataset" 并同时引 [1] 和 [5]。

---

## 2. 方法引用(⭐ 这些是最能体现方法论素养的引用)

**[6] 半监督 EM 与它的失效 —— Part 1 最重要的一条引用**
Nigam, K., McCallum, A. K., Thrun, S., & Mitchell, T. (2000). *Text classification from labeled and unlabeled documents using EM.* **Machine Learning**, 39(2–3), 103–134.

> **直接支撑论点 3**(`07_discussion_insights.md`):我们观察到 **λ=0 常常优于 λ=1**,即加入无标签数据让半监督 GMM 变差。这篇文章正是分析"生成模型误配时 EM + 无标签数据会损害分类精度"的经典工作(也提出了对无标签项加权 λ 的做法)。
> **我们代码里 `lam` 这个权重参数本身就是这篇文章的做法,所以引用既解释了结果又交代了设计出处。**

```bibtex
@article{nigam2000em,
  title   = {Text classification from labeled and unlabeled documents using {EM}},
  author  = {Nigam, Kamal and McCallum, Andrew K. and Thrun, Sebastian and Mitchell, Tom},
  journal = {Machine Learning},
  volume  = {39}, number = {2--3}, pages = {103--134}, year = {2000}
}
```

**[7] Phase augmentation —— Part 2 最重要的一条引用**
Daumé III, H. (2007). *Frustratingly easy domain adaptation.* In **Proceedings of ACL 2007**, pp. 256–263.

> **直接支撑论点 7**:`[X | 1{I}·X | 1{II}·X | 1{III}·X | phase one-hot]` 这个特征增广就是这篇文章的方法,把每个 phase 当作一个 domain。
> **引用它的价值:它把我们的 pooled 模型从"看起来投机的拼接"变成了一个有名字、有文献依据的标准方法**,并且解释了为什么 L2 正则会产生"向共享权重收缩"的分层效果。

```bibtex
@inproceedings{daume2007frustratingly,
  title     = {Frustratingly Easy Domain Adaptation},
  author    = {Daum{\'e} III, Hal},
  booktitle = {Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages     = {256--263}, year = {2007}
}
```

**[8] Label spreading(方法 G)**
Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004). *Learning with local and global consistency.* In **Advances in Neural Information Processing Systems 16 (NIPS 2003)**, pp. 321–328.

> `sklearn.semi_supervised.LabelSpreading` 实现的就是这个算法(归一化图拉普拉斯 + `alpha` 的 clamping 松弛)。我们用了 `kernel="knn"`, `alpha=0.5`, `max_iter=200`。

**[9] 自训练 / self-training 的原始形式**
Yarowsky, D. (1995). *Unsupervised word sense disambiguation rivaling supervised methods.* In **Proceedings of ACL 1995**, pp. 189–196.

> 方法 F(按置信度阈值迭代加入伪标签,5 轮)的经典出处。**可选引用** —— 如果版面紧,self-training 属于常识方法,可以不引。

**[10] k-means++ 初始化(⭐ Methods 要求写"centroids 如何初始化")**
Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding.* In **Proceedings of SODA 2007**, pp. 1027–1035.

> `KMeans(n_clusters=K, n_init=10, random_state=42)` 的默认 `init="k-means++"` 就是这个算法。**Methods 章节被明确要求说明质心初始化方式,引用这一条是最简洁的交代方式。**

**[11] Silhouette 系数**
Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.* **Journal of Computational and Applied Mathematics**, 20, 53–65.

> §1.4 聚类诊断里的 silhouette(0.16–0.31)。

**[12] Adjusted Rand Index**
Hubert, L., & Arabie, P. (1985). *Comparing partitions.* **Journal of Classification**, 2(1), 193–218.

> ARI(K=5 峰值 0.2032)。**这是支撑"聚类结构弱"最有说服力的一个指标,值得引。**

**[13] 多标签评估指标**
Zhang, M.-L., & Zhou, Z.-H. (2014). *A review on multi-label learning algorithms.* **IEEE Transactions on Knowledge and Data Engineering**, 26(8), 1819–1837.

> sample-wise F1 与 macro F1 的定义与区别。**我们的核心叙事就是"两个指标讲不同的故事"(sample-F1 被 Recruitment 主导,macro-F1 看稀有类),引一篇综述可以省掉自己定义指标的篇幅。**

---

## 3. 工具与软件

**[14] scikit-learn**
Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine learning in Python.* **Journal of Machine Learning Research**, 12, 2825–2830.

> `LogisticRegression` / `OneVsRestClassifier` / `KMeans` / `GaussianMixture` / `TfidfVectorizer` / `CountVectorizer` / `MultinomialNB` / `KNeighborsClassifier` / `LabelSpreading` / `PCA` / `TruncatedSVD` / 各类指标。

**[15] RDKit**
Landrum, G. et al. *RDKit: Open-source cheminformatics.* https://www.rdkit.org

> Morgan 指纹(`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)`)与 SMILES 解析。

**[16] Morgan / ECFP 指纹算法本身(⭐ 建议引,比只引 RDKit 更到位)**
Rogers, D., & Hahn, M. (2010). *Extended-connectivity fingerprints.* **Journal of Chemical Information and Modeling**, 50(5), 742–754.

> radius=2 的 Morgan 指纹就是 ECFP4。**引它可以支撑"为什么 radius=2 是标准选择"这一句。**

**[17] NumPy / pandas / Matplotlib**(可以合并成一条 "standard Python scientific stack" 并列引用)
- Harris, C. R., et al. (2020). *Array programming with NumPy.* **Nature**, 585, 357–362.
- McKinney, W. (2010). *Data structures for statistical computing in Python.* **Proc. SciPy 2010**.
- Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.* **Computing in Science & Engineering**, 9(3), 90–95.

---

## 4. 可选:领域背景(只在 Abstract/Discussion 需要一句"为什么这个问题重要"时引)

**[18] 临床试验失败率与成本**
Wong, C. H., Siah, K. W., & Lo, A. W. (2019). *Estimation of clinical trial success rates and related parameters.* **Biostatistics**, 20(2), 273–286.

> 提供"大多数临床试验失败、提前预测有价值"的量化背景。**一句话用途:给 Abstract 一个动机句。版面紧就删。**

---

## 5. ⭐ 引用使用建议(按版面优先级)

5 页正文下 Discussion 大约 1.3 页,引用不要铺满。**按这个优先级留:**

| 优先级 | 引用 | 用在哪里 | 为什么必须留 |
|---|---|---|---|
| **必留** | [1] HINT | Methods 的数据描述 + Discussion 的标签噪声 | 数据来源,不引是学术问题 |
| **必留** | [2] ClinicalTrials.gov | Methods 数据描述 | 同上 |
| **必留** | [6] Nigam 2000 | Discussion 论点 3 | **解释了我们最有意思的发现(λ=0 > λ=1),也是 `lam` 参数的设计出处** |
| **必留** | [7] Daumé 2007 | Methods 的 pooled 模型 | **把 phase augmentation 变成有文献依据的标准方法** |
| **强烈建议** | [14] scikit-learn | Methods 实现说明 | 一句话交代全部实现 |
| **强烈建议** | [10] k-means++ | Methods 的"质心如何初始化" | **直接对应一条硬性评分要求** |
| **建议** | [15]/[16] RDKit + ECFP | Methods 的指纹块 | 支撑 radius=2 的选择 |
| **建议** | [4] Reactome | Methods 的 drug-knowledge | 支撑 `pathway_level` 的正确用法 |
| **建议** | [3] ICD-10 | Methods 的 chapter-prefix 技巧 | 支撑层级前缀的依据 |
| 可选 | [8] 标签传播、[11] silhouette、[12] ARI、[13] 多标签指标 | Methods/Results | 标准方法,版面够就引 |
| 可删 | [5] DrugBank、[9] Yarowsky、[17] 工具链、[18] 背景 | — | 可并入正文一句话或省略 |

**最小可接受集合:[1][2][6][7][14] 五条。** 推荐集合:加上 [10][15][16][4][3],共十条。
