# Discussion 章节素材:洞察、解释与论证

Discussion 的要求是:**"insights you gained in this project, for example what gave good performance and your hypotheses on why that is the case."**
评分细则进一步要求:**不能只是盲目套算法比较;要揭示超参数/架构对性能的影响;要给合理的解释。**

下面按"可以直接写成段落"的粒度整理。每条都标了它对应的证据位置。

---

## 第一部分:Part 1 的四条核心论点

### 论点 1 ——「embedding 空间没有强聚类结构,所以纯无监督标注注定很弱」

**证据:** `03_part1_results.md` §2(silhouette 0.16–0.31 全程无峰值、inertia 无 elbow、ARI 峰值仅 0.2032 且随 K 下降、K=2 时 purity 恰等于 majority baseline 0.583)+ `p1_pca2.png`(**四个少数类 Safety/Funding/Efficacy/Other 全部挤在同一个中央区域互相纠缠,Recruitment 则弥散分布在整个右半侧**)。

> ⚠️ notebook 的 markdown 把这张图读成 "Recruitment trials form one large dense region, while Safety and Efficacy partly overlap"。**看图后确认这个描述不准确** —— Recruitment 是最**分散**的一类,而重叠的是**全部四个少数类**,不只是 Safety 与 Efficacy。报告里请用上面这个版本(详见 `06_figures.md`)。

**在分类层面的直接后果:** B0(纯簇投票)在 K=10/20 时 **Funding F1 = 0.000**;D(GMM 软标签)Funding F1 只有 0.138–0.333;G(label spreading)Recruitment F1 掉到 0.776/0.857(其他方法都 ≥0.95)。三个"纯无监督标注"家族(B0/D/G)的 macro-F1 分别是 0.684/0.617/0.482,全部**显著低于**只用 157 个标签的监督 LR(0.729–0.772)。

**解释(假设):** reason embedding 来自停止原因文本的句向量。**语义相近 ≠ 类别相同**:"terminated due to safety concerns" 和 "terminated due to lack of efficacy" 在句向量空间里很接近(都是"试验因负面结果终止"的措辞),但属于不同标签。而 "slow accrual"、"insufficient enrollment"、"PI left the institution" 都属于 Recruitment,文本差异却很大。**聚类找到的是"措辞相似性"的簇,而标签划分的是"原因类别"** —— 两种划分部分正交,这正是 ARI 只有 0.2 的含义。

**稀有类的特殊脆弱性:** Funding 只有 8 个有标签正例。在 K=10 的聚类下,这 8 个样本散落在被 Recruitment(占 64%)主导的大簇里,**没有任何一个簇以 Funding 为多数类** ⇒ 簇投票永远不会输出 Funding ⇒ F1 = 0。**这解释了为什么无监督方法在 macro-F1 上受损最严重,而 sample-F1 的损失相对温和**(sample-F1 被 Recruitment 主导,而 Recruitment 恰好是唯一聚类能抓住的类)。

### 论点 2 ——「伪标签只有在被过滤之后才有用;它起的是正则化作用,不是信息补充」

**证据:**
- 不过滤(B,cluster-then-label):val s-F1 0.816 / CV m-F1 0.735 —— **比不用无标签数据的 A(0.859/0.798)更差**。B 在 K=10 时甚至把 Safety F1 从 0.842 砸到 **0.200**。
- 按"聚类与分类器一致"过滤(C):0.876/0.769,**超过 A 的 val 分数**。
- 按"分类器自身置信度"过滤(F,self-training):0.895/0.778,同样超过 A。

**解释(假设):** 1,326 个无标签样本的伪标签里混着大量噪声。**不过滤时,噪声样本的数量(约 1,326)远大于干净样本(157),即使降权到 w_u=0.3,噪声仍然主导了损失函数** —— 伪标签不是在补充信息,而是在覆盖真标签。
过滤之后,保留下来的样本**几乎都落在决策边界远端**(高置信区),它们对系数的影响是**把决策边界推向低密度区域**(低密度分离假设),而不是重新定义类别。**这就是"伪标签作为正则化"**:它们填补了 157 个早期有标签 trial 没有覆盖到的 embedding 区域,让模型在那些区域不至于任意外推。

**一个关键的对照(诚实地写出来):** **F(self-training)完全不用聚类,分数却和 C 几乎相同**(0.895/0.769 vs 0.893/0.785 在 K=10 时)。
⇒ **本题的增益主要来自"过滤后的伪标签"这个机制本身,而不是来自聚类发现的结构。** 聚类在 C 里的作用是提供一个**与 LR 独立的第二视角**(co-training 式的一致性判据),这个作用是真实的(C 在 CV sample-F1 上达到 0.897,是全表最高之一),但它不是"聚类揭示了 reason 的结构"。**把这一点写清楚比声称"无监督学习有效"要可信得多,也更符合"不要盲目套算法"的评分要求。**

### 论点 3 ——「半监督 GMM 收敛得又快又自信,但它自信地错了 —— 生成模型误配下 EM 的经典失效」

**证据:**
- **EM 收敛良好**:70 次迭代(上限 100)、相对容差 1e-5、**91.1% 的无标签样本最大后验 ≥ 0.95**(`p1_ssgmm.png`)。
- **但分类表现差**:E(ssGMM→LR)val 0.844/0.694,CV 0.863/0.707,**低于 A**。
- **⭐ 最关键的证据:λ 的方向是反的。** npc=10 时 λ=0 → val 0.868,λ=1 → 0.724;npc=20 时 λ=0 → 0.846,λ=1 → 0.795;npc=40 时 λ=0 的 CV m-F1 是 0.779,λ=1 只有 0.631。**即"加入无标签数据让模型变差",而且越多越差。**
- **提高置信阈值没用**:conf 从 0.8 → 0.95,val 只从 0.838 到 0.844。

**解释(假设):** 我们的生成假设是"每个 label-set 对应一个高斯,所有分量共享一个协方差矩阵"。这个假设在本数据上是错的 —— 从 `p1_pca2.png` 看,**Recruitment 弥散覆盖整个右半侧、显然不是单峰**(招募失败的措辞有很多种),而**另外四个少数类全部压缩在左—中部的同一个小区域里并互相重叠**。
⇒ **两类区域的尺度相差一个数量级,"所有分量共享一个协方差"这个假设直接被图否证了。** 共用协方差只能二者取一:要么过度平滑掉少数类的紧致结构,要么低估 Recruitment 的扩散程度 —— 实际两者同时发生。
在模型误配的情况下,EM 仍然会**单调提高似然并收敛**,但它收敛到的是"在错误的模型族里最好地拟合数据"的参数,**而这个参数对分类是有害的**。无标签数据越多(λ 越大),它对参数的拉力越强,**把参数从"由 157 个真标签确定的、至少无偏的位置"拉向"生成似然最优但分类错误的位置"**。这正是 Nigam et al. (2000) 分析过的现象:*EM with unlabeled data can degrade classifier accuracy when the generative model is incorrect.*

**"自信"与"正确"的分离:** 91% 的样本后验 ≥ 0.95,意味着模型几乎把每个样本都确定地归入某个分量。但共享协方差让分量的决策边界是线性的、僵硬的,**样本被硬塞进最近的分量并获得高后验**。所以 conf 阈值过滤不掉错误 —— **错误是系统性的、而非随机的高方差**,置信度与正确率在这里几乎脱钩。
⇒ **这条论点直接满足了 Methods 对"GMM 初始化 + EM 收敛阈值 + 迭代次数"的要求,同时提供了 Discussion 需要的"为什么"。它是整份报告里理论含量最高的一段,建议重点写。**

### 论点 4 ——「验证集太小,所以我们同时看 val 和 CV;两者分歧时要诚实」

**证据:** val 只有 78 个 trial ⇒ **1 个 trial ≈ 1.3 个百分点的 sample-F1**。因此引入 **2×5 折 CV(235 个有标签样本,10 次拟合)** 作为第二套估计。

**选模的真实情况(必须坦白):**
| | val s-F1 | val m-F1 | CV s-F1 | CV m-F1 |
|---|---|---|---|---|
| A supervised LR C=0.1 | 0.8590 | 0.7291 | **0.8957** | **0.7979** |
| C K-means∩LR K=40 | 0.8718 | 0.7461 | **0.8967** | 0.7972 |
| **H ensemble(选定)** | **0.8974** | **0.8058** | 0.8900 | 0.7714 |

- **H 在 val 上两项第一,但在 CV 上两项都不是第一**;CV macro-F1 比 A **低 0.027**。
- **我们仍然选 H,理由有三条(建议原样写进报告):**
  1. **题目指定用 val 选模**("Use the performance on `part1_val.csv` to choose the model and the feature combination")。
  2. **H 的增益有机制上的解释,不只是分数巧合:** 它的优势集中在 **Funding(0.750,全表最高)和 Other(0.600,全表最高)** 两个稀有类上。平均三个错误模式不同的模型(A 判别式 / C 聚类一致性伪标签 / E 生成式伪标签)**降低了稀有类预测的方差**,而稀有类的方差正是 macro-F1 的主要误差来源。
  3. **CV 与 val 的分歧本身可以解释:** CV 的每一折只有 ~188 个训练样本(235 的 4/5),比最终模型的 235 个少;而 ensemble 的三个成员里,C 和 E 都依赖伪标签过滤,**训练集越小、过滤越不稳定**,所以 CV 低估了 H 在全量数据下的表现。
- **但要承认不确定性:** "CV score is within noise of the best" 对 sample-F1 成立(差 0.0057),**对 macro-F1 是偏乐观的**。
  ⇒ 建议的写法:*"H is the best model on the validation set, which the project specifies as the selection criterion. Its CV sample-F1 is within noise of the best (0.890 vs 0.897), though its CV macro-F1 is 0.027 lower than the plain supervised LR. We keep H because its advantage is concentrated on the two rare reasons (Funding 0.29→0.75) where averaging three differently-biased models has a clear mechanistic justification, but we note that with only 78 validation trials the ranking between A, C and H is not statistically resolved."*

**⭐ 这种坦白不会扣分,反而是 Insight 的加分项** —— 它展示了对小样本估计不确定性的理解。盲目宣称 "H is clearly the best" 更容易被质疑。

---

## 第二部分:Part 1 的特征选择论证(题目重复两次点名,务必写)

数据见 `03_part1_results.md` §6。三个要回答的问题:

**(1) 选了什么?** 只用 150 维 reason embedding。

**(2) 为什么其他特征没提升?**
**根本原因:它们描述的是"测了什么",而 reason embedding 已经编码了"为什么停"。** `criteria` / `diseases_cleaned` / `drugs` / `icdcodes` / Reactome 通路 / 药物作用,全部描述试验**设计**;而标签问的是**停止原因**,它的全部文本信息已经在 embedding 里了。
在 157 个有标签样本下,多加的 29–110 维主要增加方差。最好的单组加法(text→SVD-50)只提升 **+0.013 sample-F1 ≈ 约 1 个验证 trial**,**小于验证集噪声水平(1 trial = 1.3 分)**,因此不可信。

**(3) 为什么"全都加上"不如最好的单组?**
全加 260 维:0.8611 < text-only 的 0.8718。
**解释:四个特征组的信号高度重叠或接近于零,而噪声是独立累加的。** 当特征组之间没有互补性时,拼接只放大方差而不增加信号。这是小样本下偏差-方差权衡的直接体现。
⇒ 这恰好回答了题目的 *"why adding all features may not improve performance"*。

**(4) `start_year` 的排除:用领域知识覆盖验证分数(这是最强的一条论证)。**
它的 val sample-F1 更低(0.8526 < 0.8590),但**更重要的理由与分数无关**:题目明确说标签集偏早、test 偏晚,且 *"Recruitment is more common in early trials and Safety in later ones"*。用 `start_year` 会让模型学到"早 ⇒ Recruitment"这条**在 test 上系统性失效**的捷径。
**事后验证(很有说服力,见 `03_part1_results.md` §7):** 我们在**完全不用 `start_year`** 的情况下,模型对 test 的预测 prevalence 自发地呈现 **Safety 0.102(早期有标签)→ 0.183(最晚 test)、Recruitment 0.643 → 0.580** 的漂移,**方向与题目描述完全一致**。
⇒ **模型捕捉到的是 embedding 里真实的语义变化,而不是年份捷径。** 这是对该决定的最好验证。

**(5) 另一个被预测 prevalence 验证的理解:** Funding 的预测 prevalence(unlabeled 0.025 / test 0.032)**低于有标签集的 0.051**。这**不是漏检**,而是正确的 —— 那 40 个额外的有标签 trial 是为了覆盖稀有原因而**特意挑选**的,所以有标签集中 Funding 被人为富集。**预测值低于 0.051 恰好说明模型没有被这个采样偏差带偏。**

---

## 第三部分:Part 2 的核心论点

### 论点 5 ——「baseline 的问题是校准,不是判别能力」

**证据:** BoW LR 的 C 从 1 → 0.01:**log loss 0.9401 → 0.5918(改善 0.348)**,而 mean AUC 只从 0.6904 → 0.7374(改善 0.047)。
**baseline 的 log loss 0.9401 比直接预测各 phase base rate(≈0.675)还差。**

**解释:** 约 20,000 维特征 vs 每 phase 1,012–3,950 行,C=1 几乎不做正则化 ⇒ 训练集上线性可分 ⇒ 系数被推向大值 ⇒ 输出概率被推向 0/1。**它的排序能力勉强可用(AUC 0.69),但概率是负价值的。**
⇒ **这是整个 Part 2 的起点,也是一个可以推广的教训:当 n ≪ p 时,未正则化的 LR 的概率输出不可用,即使它的 F1 看起来正常。**

### 论点 6 ——「MNB 在 count 特征上的 log loss 是 3.65,但 AUC 正常 —— 指标选择决定结论」

**证据:** MNB counts α=1:**LL 3.6511**,AUC 0.7142,F1 0.7012。换成 TF-IDF:LL 降到 0.6524–0.7209,AUC 0.7238–0.7294。

**解释:** MNB 假设词与词独立。`criteria` 文本里的词高度相关("metastatic"、"stage IV"、"prior chemotherapy" 几乎总是同现)。**几百个相关词的对数概率被当作独立证据相加,后验被推到数值极限。** 排序(谁比谁更可能成功)仍然有效,但概率的绝对值完全不可信。
TF-IDF + `sublinear_tf` 把特征值压成分数,等效地削弱了每个词的"证据量",于是 log loss 大幅改善。
**α 的方向也印证了这一点:** α 从 0.1 → 1,log loss 从 0.7209 改善到 0.6524,但 AUC 从 0.7294 退化到 0.7238 —— **更强的平滑是在牺牲判别换校准。**
⇒ **如果只看 F1 或 AUC,MNB 是个合格的竞争者;Kaggle 用 log loss,它直接不可用。** 这是"要知道自己在优化什么"的教科书式例子,很值得写。

### 论点 7 ——「跨 phase 共享有效,但受益的是 II 和 III,不是 I —— 和直觉相反」

**证据(per-phase field-wise LR → pooled phase-augmented LR):**
AUC I **0.7638 → 0.7585(−0.005)**;AUC II 0.7246 → 0.7365(+0.012);AUC III 0.7758 → 0.7886(+0.013);mean LL 0.5756 → 0.5665。

**机制:** `[X | 1{I}·X | 1{II}·X | 1{III}·X | phase one-hot]`(Daumé III 2007 的特征增广)。L2 正则把 phase-specific 副本的权重向 0 收缩,**等价于向共享权重收缩** —— 这就是分层收缩。

**⚠️ 与直觉相反的结果:** phase I 训练数据最少(1,012 行),按"借力"的逻辑它应该受益最多,**实际上它略微变差了**。
**假设(值得写,这是一个有内容的推测):** **phase I 的"成功"定义与 II/III 差异最大** —— phase I 的主要终点通常是安全性/耐受性/剂量确定,而 II/III 看疗效。因此 **phase I 与 II/III 之间的"domain 距离"最大**,共享权重对它不是助力而是干扰;而 II 和 III 的成功判据更接近,互相借力更有效。
这个假设可以被 phase III 的最大增益(+0.013)佐证:III 的数据量也不小(3,075),它受益说明**增益来源不是"样本量不足被补足",而是"相邻 phase 的信号确实可迁移"**。

**合规性(引题目原文):** *"Success means something different in each phase, so either train a separate model per phase, or train a single model that uses the phase."* ⇒ pooled 模型用 phase 作为特征,属于题目明确许可的第二种做法。

### 论点 8 ——「三类领域知识特征全部无效 —— 这是最值得讨论的负面结果」

**证据(相对 field-wise TF-IDF C=1 基准 LL 0.5756 / AUC 0.7547):**
+ Reactome pathways/targets/actions:0.5757 / 0.7533(无变化)
+ Morgan fingerprints:0.5754 / 0.7530(无变化)
+ start_year:0.5778 / 0.7489(略差)

**三层解释:**
1. **覆盖率:只有 45.7% 的 train 行有匹配药物。** 另一半的指纹是全零、通路/靶点是空串 —— 对一半数据零信息,但对所有行都增加维度。而且**缺失不是随机的**:生物制剂(抗体等)系统性匹配不上,所以"有无匹配药物"混杂了"药物模态"这个混淆因素。
2. **语义:化学结构和分子通路回答"这个药在生物学上做什么",而标签问"这个试验有没有达到主要终点"。** 后者主要由**试验设计**决定(终点选得对不对、入排严不严、招募可行不可行、统计功效够不够)。**分子结构与试验设计几乎正交。**
3. **冗余:`drugs` 字段里的药名本身已经是最强的药物特征。** 同一个药在多个试验中出现,LR 直接从药名学到它的历史成败,比从 1024 位指纹里重新推断结构-活性关系直接得多。**指纹只在需要对未见过的新药泛化时才有价值,而 test 的药物与 train 大量重叠。**

⇒ **这条论点的价值在于它是一个被充分解释的负面结果。** 题目要求"讨论为什么某些特征没能提升性能",而我们对 Part 1 和 Part 2 给出的是**同构的答案:描述"测了什么"的特征,对"结果如何"几乎没有增量信息。** 两个部分的结论互相印证,是报告里一条很好的贯穿主线。

### 论点 9 ——「ensemble 的收益在噪声内,所以我们选了更简单的模型」

最好的 ensemble 0.5639 vs 单模型 pooled LR 0.5665,**差 0.0026**,在 1,013 行验证集上不可靠。
**我们据此保留单模型。** 这是一个应该写出来的决定:**比较了并拒绝了更复杂的方案**,而不是默认"ensemble 一定更好"。
(有意思的对照:**Part 1 我们采用了 ensemble,Part 2 我们拒绝了 ensemble。** 差别在于 Part 1 的 ensemble 有机制上的理由——救稀有类——且增益幅度(macro-F1 +0.03 以上)大于噪声;Part 2 的 ensemble 只有 0.0026 的增益且无机制解释。**同一个技术在两个任务上被不同对待,并且给出了判据,这正是"不盲目套算法"的体现。**)

---

## 第四部分:Part 2 错误分析的解读(⭐ 评分细则点名三条)

> 评分细则原文:**"Part 2 要讨论不同 reason 类别间错误率的差异,并考虑类别规模和预测 reason 标签本身可能有错。"**

完整数据见 `05_part2_results.md` §4–5。四条要写的内容:

### (a) 不同 reason 类别的错误率差异与解释

| 组 | 假阳性率表现 | 解释 |
|---|---|---|
| **No recorded stopping reason**(最大组:28/144/82,占各 phase 失败试验的 44%/54%/67%) | baseline 下 phase I 最高(0.571) | **这些试验跑完了全程但没达到主要终点,没有提前停止,所以不在 Part 1 里。** 它们的文本和成功试验没有区别 —— 都是设计完整、执行到底的试验。**结构性地最难。** |
| **Recruitment**(phase II n=88) | **改进幅度最大:0.352 → 0.239** | 招募失败的试验往往规模小、单中心、入排标准过窄,**这些特征直接写在 `criteria` 文本里**("must have failed ≥3 prior therapies"、罕见基因型要求)。1–2gram 捕捉到了这种短语模式。**这是唯一一条完整的因果链:特征里有信号 ⇒ 模型学到 ⇒ 错误率下降。** |
| **Phase III 的 Safety / Efficacy** | 两个模型下都高:0.57–0.78 | phase III 是大规模确证性试验,设计精良、入排规范,**文本与成功的 phase III 无法区分**。失败来自**药物本身的效果**(疗效不足或安全性信号),而文本完全不含这个信息。**这是特征层面的根本局限,不是模型容量不够 —— 再优化分类器也救不了这组。** 需要的是中期分析数据和不良事件报告,数据集里没有。 |

### (b) ⭐ phase III 整体假阳性率上升(0.496 → 0.569)但所有指标改善 —— 这不是矛盾

phase III 的 base rate 是 0.653。改进模型**更好地校准到这个先验**,于是整体更倾向预测成功。代价是失败试验上的假阳性更多,收益是成功试验上的召回更高(而成功试验占 65%)。
⇒ **这正是题目要求的 trade-off 讨论。** 把它和阈值表(`05_part2_results.md` §6)放在一起:**如果任务是"筛掉会失败的试验",应该用更高的阈值;如果任务是"给出校准的成功概率"(Kaggle 的 log loss),就该保持当前设定。**
配套数字:阈值 0.5 → 0.7,**phase II 的假阳性率从 0.283 降到 0.083(3.4 倍)**,但 **F1 从 0.645 掉到 0.504**,TPR 从 0.627 掉到 0.369。**log loss 和 AUC 完全不受阈值影响。**
另一观察:**phase III 在所有阈值下 FPR 都显著高于 I/II**,这是 base rate 0.653 的直接后果 —— 统一的 0.5 阈值对 phase III 更宽松。**如果要均衡各 phase 的 FPR,应该用 per-phase 分位数阈值**(可作 future work)。

### (c) ⭐ 类别规模(题目点名)

- **Funding:phase I n=4、phase II n=4、phase III n=0。** 1 个 trial 翻转改变 0.25–0.5。表中 Funding 从 0.500/0.250 掉到 0.000/0.000 看似巨大改善,**实际只是 2–3 个 trial 的随机变动,不可解读。**
- phase I Efficacy(n=2)从 0.500 → 1.000 —— 1 个 trial 的变化,无意义。
- phase I Other(n=3)、phase III Safety(n=7)/Efficacy(n=9)/Other(n=10) 也在不可靠区间。
- **建议在报告里明确划线:n < 10 的组不做解读。** 样本量够的只有:No-reason(28/144/82)、phase II Recruitment(88)、phase II Other(22)、phase I Recruitment(23)、phase III Recruitment(21)、phase II Safety(13)、phase II Efficacy(11)。
- **⚠️ phase III 的 Funding 组 n=0 ⇒ 表里是 NaN、图里是空柱子,必须加注说明。**

### (d) ⭐⭐ 预测 reason 标签本身可能有错(题目点名)

- `reason_lookup` 里 **1,326 个是 Part 1 的预测,235 个是真标签**。逐组数字在 `05_part2_results.md` §4 的 `true / predicted` 列。
- 最大的组几乎全是预测:**phase II 的 Recruitment 组 88 个中只有 12 个真标签、76 个预测**;phase I Safety 组 9 个中 8 个预测;phase III Efficacy 组 9 个全是预测。
- Part 1 的验证 sample-F1 ≈ 0.90 ⇒ **大约 10% 的 trial 被分到了错误的 reason 组**。
- **⭐ 关键在于这个误差不是均匀分布的:**
  - Part 1 对 **Recruitment 的 F1 是 0.99** ⇒ **最大的组(Recruitment)的分组几乎可信,这是一个重要的缓解因素。**
  - 但 Part 1 对 **Funding(0.75)和 Other(0.60)的 F1 低得多** ⇒ **恰恰是样本量最小的那几组,分组也最不可靠。**
- ⇒ **结论:两种误差来源(小样本 + 标签不确定)叠加在同一批组上。大组(Recruitment、No-reason)的结论可信;小组(Funding、Other)双重不可解读。** 这个"误差叠加"的观察是整段里最有价值的一句。

---

## 第五部分:时间漂移(⭐ 题目明确要求讨论 val→test 差距)

> 题目:*"Expect lower performance on the test set than on the validation set, since the test trials are from later years; discuss this gap in your report."*

我们没有只是"讨论",而是**用已有数据构造了一个代理实验去量化它**:train∪val 按 `start_year` 切成 early(≤2008,6,326 行)和 late(≥2010,2,290 行),留 2009 作缓冲,在 early 上训练、late 上评估。**完全不需要 test 标签。**

| | random val | temporal (early→late) | 变化 |
|---|---|---|---|
| improved mean AUC | 0.7612 | 0.6948 | **−0.066** |
| baseline mean AUC | 0.6904 | 0.6173 | **−0.073** |
| improved mean LogLoss | 0.5665 | 0.5672 | **+0.0007(几乎不变)** |
| baseline mean LogLoss | 0.9401 | 1.0184 | **+0.078(进一步恶化)** |

成功率本身也漂移:**early 0.522 → late 0.655**。

**三条结论:**

1. **预期 Kaggle 的 AUC 比验证集低约 0.07。** 这是对题目那句话的量化回答。下降来自词汇漂移(新药名、新疗法、新的 criteria 写法)和适应症构成变化。
2. **⭐ 改进模型的 log loss 在漂移下几乎不退化(0.5665 → 0.5672),而 baseline 恶化到 1.0184。**
   **这是 Part 2 最强的一条结论:正则化带来的好校准在分布漂移下依然成立,而过度自信的模型在漂移下被惩罚得更重。**
   机制:判别能力下降(AUC −0.066)本应推高 log loss,但**后期试验的 base rate 更高(0.655 vs 0.522),"预测成功"这个方向变简单了**,两个效应抵消;模型通过 **phase 截距项**部分吸收了先验变化。baseline 没有这种抵消 —— 它的概率本就极端,漂移让更多极端预测变错。
3. **向先验收缩没有帮助:两套评估的最优 s 都是 1.0**,任何收缩都让 log loss 变差,**而且随机划分和时间划分给出完全一致的结论**(0.5665/0.5682/0.5724/0.5787 vs 0.5672/0.5683/0.5717/0.5772)。
   ⇒ 正则化后的 LR 已经校准得很好,不需要事后收缩。**这个负面结果的价值在于它证明了论点 5 的判断:问题已经在正则化阶段解决了,不需要第二道防线。**

**此外(`class_weight="balanced"`,另一个"试了但不用"的改动):** mean AUC 0.7612 → 0.7609,mean LL 0.5665 → 0.5685,三个 phase 的 F1 都略降。原因是各 phase base rate 本来就是 0.477–0.653、**接近平衡**;balanced 加权是为严重不平衡设计的,这里只是把边界推偏,打乱了已经不错的校准。
⇒ 题目明确说 **"Even if the change does not improve the F1 score, discuss the results"**,所以这两个负面改动(shrinkage、class weighting)都必须写。

---

## 第六部分:可写进 Discussion 结尾的「局限与 future work」

1. **标签噪声是不可约误差的一部分。** 题目说 HINT 的标签 *"we corrected 7 labels that contradicted the trial's ClinicalTrials.gov record, but some label noise likely remains"*。这部分解释了为什么 ROC-AUC 停在 0.76 左右而不是更高。
2. **Part 2 的信息瓶颈在特征,不在模型。** phase III 的 Safety/Efficacy 失败来自药物效果,文本不含这个信息(论点 8 和错误分析 (a) 共同支持)。**要突破需要试验中期数据,不是更复杂的分类器。**
3. **Part 1 预测的多标签率偏高:** 提交中 50/312 = 16.0% 是多标签,而训练集真实多标签率只有 10/157 = 6.4%。ensemble 的概率平均把三个模型都"略微倾向"的第二标签推过了 0.5。**调高第二标签的阈值(或要求概率差距足够大)是一个有依据的改进方向。**
4. **未尝试的方向(诚实列出):** DBSCAN(题目建议清单里有,我们只做了 K-means 和 GMM)、Part 1 的 KNN(题目建议清单里有)、RDKit 的 Tanimoto 相似度与分子描述符(MolWt / MolLogP)、per-phase 分位数阈值、field 权重的网格搜索。
5. **转导设定的说明(⭐ 建议主动写,避免被误解):** 所有半监督方法的无标签池 `XU` 包含 **test 的 `emb_*` 特征**(但不含任何 test 标签),所以 Part 1 的 test 预测是**转导式(transductive)**的。这是标准的半监督设定,**不是题目禁止的"反推 test 标签"或"从网上找完整标注数据"**。由此产生的不对称:`XU` 不含 val、也不含 CV 的留出折,所以 **val/CV 分数是归纳式的,而 test 预测是转导式的** —— 实际 test 表现可能略优于验证估计。
