# 数据集事实(写 Methods / Results 开头时可直接引用)

两类来源分开标注:**[题目]** = PDF 原文给定;**[实测]** = notebook 运行输出。

## 数据来源 [题目]

- 基于 **HINT benchmark**(Fu et al., *Patterns*, 2022)和 **ClinicalTrials.gov** 的临床试验数据。
- 数据通过两个 Kaggle 竞赛分别提供。
- Part 1 竞赛提供 4 个 CSV:`part1_train.csv`、`part1_val.csv`、`part1_test.csv`、`pathway_catalog.csv`。
- Part 2 竞赛提供 3 个 CSV:`outcome_train.csv`、`outcome_val.csv`、`outcome_test.csv`。

## Part 1 数据

### 规模与划分 [题目]

- 共 **1,561 个 trial**,全部是**提前停止**(terminated / withdrawn / suspended)**且有记录的停止原因**的试验,涵盖所有 phase。
- 大部分 trial 的原因来自 ClinicalTrials.gov 的 `why_stop` 字段;**有 42 个 trial 没有记录原因,由课程助教从记录的其他部分(如 results section)恢复并标注。**
- **按 start year 划分**:`part1_test.csv` 是**最晚的 312 个 trial(2010–2016)**;更早的 trial 随机划分为 `part1_train.csv`(1,171)和 `part1_val.csv`(78)。

### 标签遮蔽方式 [题目] —— 这是 Part 1 的核心设定

- `part1_train.csv` 中 **157 个 trial 标签可见**,构成方式是:
  - 按 `start_year` **最早的 10%(117 个 trial)**
  - **外加 40 个** 为了包含更多稀有原因(Safety、Funding、Efficacy、Other)而**特意挑选**的 trial
- 剩余 **1,014 个 trial 的全部五个标签被遮蔽为 `-100`**。
- `part1_val.csv` 标签全部可见;`part1_test.csv` 标签全部隐藏。

> **题目原文的关键提示(直接决定 Methods 的设计论证):**
> "Because the labeled trials are mostly early and the test trials are the latest, the frequency of each reason differs between the labeled, unlabeled, and test trials (for example, **Recruitment is more common in early trials and Safety in later ones**); consider this when designing your method."
>
> 这句话是我们**不使用 `start_year` 作为特征**的直接依据 —— 有标签的样本集中在早期,模型会学到"早期 ⇒ Recruitment"的关联,而这个关联在 test 上不成立。

### 列构成 [题目]

`ID`、`start_year`、150 维 reason embedding(`emb_0` … `emb_149`)、5 个二值 reason 标签列(`part1_test.csv` 中不提供)、`criteria`(入排标准)、`diseases_cleaned`、`drugs`、`icdcodes`,以及 drug-knowledge 特征。

### Reason embedding 的生成方式 [题目] —— Methods 里必须交代

> "The reason embeddings were computed from the stopping-reason text with a **pretrained sentence-embedding model**, reduced to **150 dimensions with PCA**, and **randomly rotated**, so **individual dimensions have no meaning**; the embedding values are small, so **standardize them before use**."

三个推论,都可以写进报告:
1. 单个维度无可解释性 ⇒ 不做特征选择/不解读系数。
2. 已经过 PCA ⇒ 后续再做 PCA 只是为了降维去噪,不是为了去相关。
3. 数值很小 ⇒ **必须标准化**(我们用 `StandardScaler`,统计量来自 train 文件的 labeled+unlabeled 全体)。

**原始 `why_stop` 文本不提供。** 原文还说:"You may only use a subset of available features to maximize the model performance."(即允许只用部分特征。)

### Drug-knowledge 特征 [题目]

- 按**药名精确匹配**到一个 DrugBank 衍生的**小分子药物**表。因此**生物制剂(如抗体)和用别名列出的药物匹配不上**。
- **约 56% 的 Part 1 trial 至少匹配到一个药物**,其余的药物列为空。
- 每列用分号分隔该 trial 所有已匹配药物的值:
  - `matched_drugs`:匹配上的药名。
  - `drug_targets`:药物结合的蛋白靶点,以**基因符号**给出。
  - `drug_pathways`:**Reactome ID**。药物不直接作用于通路 —— 它结合一个或多个靶点,每个靶点参与 Reactome 整理的通路;**一个药物的通路是其所有靶点通路的并集**。
  - `drug_actions`:药物如何影响靶点(如 inhibitor、agonist)。**动作是在该 trial 所有已匹配药物的所有靶点上汇总的,所以"某个具体动作对应某个具体靶点"的关联不可得。**
  - `drug_smiles`:SMILES 字符串表示的化学结构。同一分子可以写成多个合法 SMILES。
- `pathway_catalog.csv`:`reactome_id` → `pathway_name`、`pathway_level`,以及该通路中的药物数和靶点数。
- **`pathway_level`**:Reactome 的层级深度。**level 0 是 29 个最宽泛的类别**(如 Signal Transduction、Metabolism、Immune System),越深越具体,**最深是 level 9**。属于某个具体通路的靶点也会被列在所有包含它的更宽泛通路下。**`pathway_level` 描述的是通路而非药物,且是序数深度而非测量量。**(⇒ 不应把它当数值特征直接用。)

### 实测数值 [实测]

| 项 | 值 |
|---|---|
| `reason_train.shape` | **(1171, 166)** |
| `reason_val.shape` | **(78, 166)** |
| `reason_test.shape` | **(312, 161)** —— 少 5 列即 5 个标签列 |
| labeled trials | **157** |
| unlabeled trials | **1014** |
| 半监督方法使用的 unlabeled pool | **1326** = 1014 遮蔽的 train + 312 test |
| labeled ∪ val(CV 用的全量标签数据) | **235** |

### Reason 的类别分布 [实测]

| reason | labeled train (157) | val (78) |
|---|---|---|
| Safety | 0.102 | 0.115 |
| Funding | **0.051** | 0.051 |
| Efficacy | 0.108 | 0.179 |
| Recruitment | **0.643** | 0.641 |
| Other | 0.159 | 0.128 |

- Funding 在 157 个有标签 trial 中只有 **8 个正例**(0.051 × 157 ≈ 8)。这是 `class_weight="balanced"` 的直接动机,也是 macro-F1 波动大的原因。
- **每个 trial 的标签数**:labeled `{1: 147, 2: 10}`;val `{1: 69, 2: 9}`。
  ⇒ **几乎每个 trial 恰好只有一个 reason**(147/157、69/78)。这是 `at_least_one` 后处理和"按 label-set 建 GMM 分量"两个设计的依据。
- Recruitment 占约 64%,是绝对主类。**全预测 Recruitment 的 majority-class purity = 0.583**(235 个有标签 trial 上)。

## Part 2 数据

### 规模与划分 [题目]

- 包含三个 phase 的试验,二值 outcome 标签表示该试验**是否达到主要终点**(label=1 达到,label=0 未达到)。
- **每行是"一个 trial 在一个 phase"**:combined-phase 研究(如 Phase I/II)在每个 phase 各出现一次,**标签相同**。
- `outcome_train.csv`(**8,037 行**)和 `outcome_val.csv`(**1,013 行**)包含 **start 年份不晚于 2013** 的 trial,**按 trial 随机划分**,所以同一个 trial 不会同时出现在两边。
- `outcome_test.csv` **1,500 行(每 phase 500 行)**,来自**更晚开始的 trial(2013–2022)**;**它们不出现在任何其他文件中,包括 Part 1 的文件**。

> ⚠️ **当前 notebook 的 2.5 节和时间稳健性段落写的是 "test trials start in 2014–2021",与题目的 2013–2022 不一致,报告里需改正。** 详见 `09_issues_to_address.md`。

### 列构成 [题目]

`ID`(trial ID + phase,如 `TR00042_II`)、`trial_id`、`phase`(I/II/III)、`phase_detail`(如 "phase 1/phase 2")、`criteria`、`diseases_cleaned`、`drugs`、`icdcodes`、Part 1 中描述的 drug-knowledge 列、`start_year`、二值 `label`(test 中无)。

### 标签质量 [题目] —— 值得写进 Discussion

> "The labels come from the HINT benchmark; **we corrected 7 labels that contradicted the trial's ClinicalTrials.gov record, but some label noise likely remains.**"

⇒ 存在不可约误差(irreducible error),是 AUC 停在 0.76 左右的部分解释。

### 实测数值 [实测]

| 项 | 值 |
|---|---|
| `outcome_train.shape` | **(8037, 15)** |
| `outcome_val.shape` | **(1013, 15)** |
| `outcome_test.shape` | **(1500, 14)** |
| train ∪ val(最终重训用) | **9,050 行** |
| train 中有匹配药物的行比例 | **0.457**(与 Part 1 的 ~56% 不同,Part 2 更低) |

### 各 phase 规模与 base rate [实测] —— Results 表格的基准

| phase | train 行数 | train success rate | val 行数 | val success rate |
|---|---|---|---|---|
| I | 1,012 | 0.566 | 149 | 0.570 |
| II | 3,950 | 0.477 | 501 | 0.471 |
| III | 3,075 | 0.653 | 363 | 0.661 |

几个可用于论证的点:
- **Phase I 训练数据最少(1,012 行)**,这是 pooled phase-augmented 模型"让 Phase I 向 II/III 借力"的直接动机。
- **Phase II 的 base rate 最接近 0.5(0.477),也是三个 phase 中最难的**(所有模型在 II 上 F1 和 AUC 都最低)。
- **Phase III base rate 最高(0.653)**,因此它的 F1 天然偏高,而"预测成功"的倾向也更强 —— 这解释了改进模型在 Phase III 上假阳性率反而上升的现象。
- 预测 base rate 的 log loss 基准:phase I ≈ 0.685、II ≈ 0.692、III ≈ 0.648(由 $-[\pi\ln\pi+(1-\pi)\ln(1-\pi)]$ 算得,均值 ≈ 0.675)。**baseline 的 0.9401 比这个还差**,是报告里很有力的一句话。

## RDKit 使用说明 [题目]

题目专门用一页介绍 RDKit,并给出了示例代码。要点:
- `Chem.MolFromSmiles` 解析 SMILES,无效串返回 `None`。
- **Morgan(circular)fingerprint** 是定长二值向量(如 1024 位),每位表示分子是否含某个小子结构。结构相似的分子指纹相似,可直接作为 sklearn 特征。
- Descriptors:`Descriptors.MolWt`(分子量)、`Descriptors.MolLogP`(亲脂性)。
- **Tanimoto 相似度**(0–1)衡量两个指纹的结构相似程度,可用作聚类或近邻方法的相似度度量。
- 一个 trial 可能有多个匹配药物(`drug_smiles` 中分号分隔)或没有,**所以需要自己决定如何跨药物合并指纹、以及如何处理无匹配药物的 trial**。题目示例取**逐元素最大值**(任一药物含该子结构则该位为 1),无匹配药物则取全零向量,并说明"其他选择也是可以的"。

> 我们的实现与题目示例一致(radius=2, fpSize=1024, 跨药物取 max, 无药物取全零),额外做了一步按行归一化以匹配 TF-IDF 的尺度,并按 SMILES 字符串缓存。**题目没有要求用 Tanimoto 或 Descriptors,我们也没用 —— 如果要补 Creativity,这是一个低成本方向。**
</content>
