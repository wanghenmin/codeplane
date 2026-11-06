# Part 2 方法细节(Methods 章节素材)

## 0. 合规声明(⭐ 报告里必须有一句)

> **"Reason embeddings and reason labels should be used only for error analysis, not as predictive inputs to the binary classifier."**

notebook 2.3 节开头已明确写了 "Reason labels and embeddings are **never** used as inputs here"。
Part 2 的所有特征只来自 `criteria`、`diseases_cleaned`、`drugs`、`icdcodes`、drug-knowledge 列、`phase`、`start_year`。**reason 标签只在 2.2 / 2.4 的错误分析里出现。** 报告里要显式声明这一点。

同样要声明:**`ID` / `trial_id` 只用于把 Part 2 的失败试验匹配到 Part 1 的 reason 标签,从未作为预测特征。**

## 1. 通用文本与评估函数

```python
text_columns = ["criteria", "diseases_cleaned", "drugs", "icdcodes"]
phases = ["I", "II", "III"]
trial_text = lambda df: df[text_columns].fillna("").agg(" ".join, axis=1)

def metrics(y, p, thr=0.5):
    return {"F1": f1_score(y, p >= thr, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p),
            "Log Loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}
```

- 四个指标与题目要求一致(F1 / ROC-AUC / PR-AUC,以 label=1 为正类),**额外报告 log loss 因为 Kaggle 的评分就是 log loss**。
- **log loss 计算前把概率 clip 到 [1e-6, 1-1e-6]**,否则 MNB 之类的过度自信模型会给出 inf。
- `run_phasewise(name, make, train_df, eval_df)` 对三个 phase 分别训练+评估,然后记录各 phase 指标和三 phase 均值。**因为 test 的三个 phase 行数相同(各 500),所以 "mean log loss" 正是 Kaggle 分数的估计。**

## 2. Baseline(starter code,2.1 节)

```python
for phase in ["I", "II", "III"]:
    vectorizer = CountVectorizer(binary=True, min_df=3, max_features=20000)
    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
```
即**每个 phase 一个独立模型**、**二值 bag-of-words**、**C=1(默认)**、无停用词、无 TF-IDF。

👉 **诊断(这是 Part 2 叙事的起点):** 约 20,000 个特征 vs 每 phase 只有 1,012–3,950 行训练数据,C=1 几乎不做正则化。结果是模型**严重过度自信**:log loss 0.9401,**比直接预测 base rate(≈0.675)还差**。即 baseline 的概率输出几乎没有可用的校准性,即使它的 F1 看起来还行(0.6962)。

## 3. 候选模型(2.3 节)——Creativity 的覆盖面

### 3.1 文本表示(题目列出的三种全部覆盖)

| 表示 | 配置 |
|---|---|
| **Bag of words** | `CountVectorizer(binary=True, min_df=3, max_features=20000)` |
| **TF-IDF unigram** | `TfidfVectorizer(min_df=3, max_features=50000, sublinear_tf=True, stop_words="english")` |
| **N-gram (TF-IDF 1–2gram)** | `TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=100000, sublinear_tf=True, stop_words="english")` |
| counts(供 MNB) | `CountVectorizer(min_df=3, max_features=20000)` |

`sublinear_tf=True` 即用 `1+log(tf)` 替代原始词频 —— 对 `criteria` 这种长文本很重要,防止长文档里重复出现的词被过度加权。

### 3.2 分类器(题目列出的三种全部覆盖)

| 分类器 | 配置与扫描范围 |
|---|---|
| **Binary Logistic Regression** | `LogisticRegression(C, max_iter=3000, solver="liblinear")`;BoW 上扫 C ∈ {1, 0.01, 0.03, 0.1, 0.3},TF-IDF uni 与 1–2gram 上扫 C ∈ {1, 3, 10} |
| **Multinomial Naive Bayes** | counts 上 α ∈ {1, 10};TF-IDF 上 α ∈ {0.1, 0.3, 1} |
| **KNN** | `KNeighborsClassifier(k, metric="cosine", weights="distance")` on TF-IDF unigram,k ∈ {25, 75, 150} |

- **MNB 用的是非负的 count / TF-IDF 特征**,符合题目提示。
- **kNN 用余弦距离 + 距离加权**(而不是欧氏/均匀),因为高维稀疏 TF-IDF 上欧氏距离几乎无意义。

### 3.3 Field-wise TF-IDF(自己设计的特征工程)

```python
class FieldFeaturizer:
    def __init__(self, weights=None, drug=0.0, year=0.0, fp=0.0):
        self.w = weights or {"criteria": 1.0, "diseases": 0.5, "drugs": 0.5, "icd": 0.5}
```

**动机:** baseline 把四个字段拼成一个字符串,于是 `criteria`(几百词)完全淹没了 `drugs`(几个词)和 `icdcodes`。**分字段各建一个 TF-IDF、再加权拼接**,让短字段也有话语权。

每个字段的 vectorizer 设定:

| 字段 | vectorizer | 权重 |
|---|---|---|
| `criteria` | `tfidf_bi`(1–2gram, min_df=3, max_features=100000, sublinear_tf, English stop words) | **1.0** |
| `diseases_cleaned` | `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` | 0.5 |
| `drugs` | `TfidfVectorizer(min_df=2, sublinear_tf=True)` | 0.5 |
| `icdcodes` | `TfidfVectorizer(min_df=2, token_pattern=r"[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?", lowercase=False)` | 0.5 |
| `drugk`(可选) | `TfidfVectorizer(min_df=3, token_pattern=r"[^ ]+", lowercase=False, sublinear_tf=True)` | `drug` 参数(用 0.3) |

**权重 1.0 / 0.5 是先验选择(criteria 信息量最大),没有做权重搜索** —— 报告里如实说明即可,这是一个合理但未调优的选择。

#### ⭐ ICD chapter-prefix 技巧(值得单独写,是一个有想法的小设计)

```python
def icd_text(df):  # E11.65 -> "E11 E11.65"
    return df.icdcodes.fillna("").str.replace(r"([A-Z][0-9]{2})\.", r"\1 \1.", regex=True)
```
**ICD 码是层级的**:`E11.65` 和 `E11.9` 都是 2 型糖尿病,但作为独立 token 它们毫无关系。把三字符的 chapter 前缀**额外**作为一个 token 加进去,**同一疾病大类的不同细分码就共享一个特征**。这在 `min_df=2` 的稀疏 ICD 空间里能显著提高泛化。
配套的 `token_pattern=r"[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?"` 和 `lowercase=False` 保证 ICD 码不被默认分词器切碎。

#### drug-knowledge 的文本化

```python
def drug_text(df):
    return (df.drug_pathways.fillna("").str.replace(";", " ") + " "
            + "TGT_" 前缀的 drug_targets + " "
            + "ACT_" 前缀的 drug_actions(空格换成下划线))
```
**用 `TGT_` / `ACT_` 前缀把三种不同语义的实体放进同一个词袋而不互相碰撞**,Reactome ID 本身已经是唯一字符串所以不用前缀。`ACT_` 里把空格换成下划线,让 "enzyme inhibitor" 保持为一个 token。

#### ⭐ RDKit Morgan 指纹

```python
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
# 一个 trial 的指纹 = 其所有可解析药物指纹的逐元素 max;无匹配药物 → 全零
F / np.sqrt(np.maximum(F.sum(1, keepdims=True), 1))     # 按行归一化,匹配 TF-IDF 的尺度
```
- **radius=2, 1024 bits**,与题目示例一致。
- **跨药物取逐元素最大值**(= 按位 OR:任一药物含该子结构即置 1),与题目示例一致。
- **无匹配药物 → 全零向量**,与题目示例一致。
- 两处自己的处理:(1) **按 SMILES 字符串缓存** `_fp_cache`,避免重复解析;(2) **按行做 L2 式归一化**(除以 `sqrt(非零位数)`),让指纹块的尺度与 TF-IDF 块可比,否则一个 1024 位的 0/1 块会在加权拼接里压倒 TF-IDF。
- `Chem.MolFromSmiles` 返回 `None` 的无效 SMILES 被跳过;`RDLogger.DisableLog` 关掉警告。
- 块权重 `fp=0.3`。

### 3.4 ⭐⭐ Pooled phase-augmented LR(最终选定的模型)

这是 Part 2 的方法核心,**Methods 章节应该重点展开**。

```python
def phase_augment(X, ph_arr):
    """[shared | phase-I copy | phase-II copy | phase-III copy | phase one-hot]"""
    blocks = [X] + [sp.diags((ph_arr == p).astype(float)) @ X for p in phases]
    onehot = sp.csr_matrix(np.column_stack([(ph_arr == p).astype(float) for p in phases]))
    return sp.hstack(blocks + [onehot]).tocsr()

class PooledPhaseLR:
    def fit(self, df):
        self.f = FieldFeaturizer(**self.fk).fit(df)
        self.m = LogisticRegression(C=self.C, max_iter=5000, solver="liblinear",
                                    class_weight=self.cw).fit(
            phase_augment(self.f.transform(df), df.phase.values), df.label)
```

**结构:** 特征矩阵变成 `[X | 1{phase=I}·X | 1{phase=II}·X | 1{phase=III}·X | phase one-hot]`,维度是原来的 4 倍 + 3。

**这正是 Daumé III (2007) 的 "Frustratingly Easy Domain Adaptation" 特征增广**,把每个 phase 当作一个 domain。等价地,phase p 的有效权重是 `w_shared + w_p`:

- **共享块 `X`** 学习"所有 phase 共通的成功/失败信号"(例如罕见病、模糊的入排标准)。
- **每个 phase 的副本** 学习"该 phase 特有的偏离"。
- **phase one-hot** 吸收各 phase 不同的 base rate(0.566 / 0.477 / 0.653),作为截距项。
- **L2 正则(C=1)在这个结构下是关键**:它把每个 phase 的副本权重向 0 收缩,**也就是向共享权重收缩**。数据多的 phase(II、III)可以支撑较大的 phase-specific 偏离;**数据少的 phase I(仅 1,012 行)会被拉回共享权重,相当于自动从 II/III "借力"**。这就是分层收缩(hierarchical shrinkage)的效果,而且是用一行稀疏矩阵代数实现的。

**合法性(可引题目原文):** 题目明确说 *"Success means something different in each phase, so either train a separate model per phase, or train a single model that uses the phase."* ⇒ **pooled 模型用了 phase 作为特征,属于后一种做法,完全合规。** 报告里引这句很有价值,因为它把一个看起来"投机"的设计变成题目明确许可的选项。

- 扫 C ∈ {0.3, 1, 3}。
- 实现上有个 `pooled_cache`,因为 `run_phasewise` 会对三个 phase 各调一次 `make`,而 pooled 模型只需拟合一次。

### 3.5 Ensemble

```python
def make_ens(*makers, weights=None):
    return lambda a, b, ph: np.average([m(a, b, ph) for m in makers], axis=0, weights=weights)
```
三个组合:
- pooled LR + per-phase TF-IDF 1–2gram LR(C=3),等权
- pooled LR + kNN k=75,权重 [0.7, 0.3]
- pooled LR + MNB TF-IDF α=0.3,权重 [0.7, 0.3]

**结论是 ensemble 只比单模型好 ~0.002 log loss(在噪声内),所以最终保留单模型。** 这是一个诚实的负面结果,报告里应该写 —— 它说明我们**做了比较并据此拒绝了更复杂的方案**,而不是默认"ensemble 一定更好"。

## 4. 错误分析方法(2.2 / 2.4 节)⭐ 题目有详细规定

```python
def reason_error_table(results, prob_col="probability", thr=0.5):
    failed = results[results["label"] == 0].copy()           # 只看真实失败的试验
    failed["pred_success"] = (failed[prob_col] >= thr).astype(int)
    m = failed.join(reason_lookup, on="trial_id")            # 用 trial_id 匹配 Part 1 的 ID
    for ph in phases:
        f = m[m.phase == ph]
        assert f.trial_id.is_unique                          # ⭐ 保证一个 phase 内不重复计数
        groups = {r: f[f[r] == 1] for r in labels}           # 一个 trial 可同时进多组
        groups["No recorded stopping reason"] = f[f.source.isna()]
```

逐条对应题目的要求:

| 题目要求 | 实现 |
|---|---|
| 只看 `outcome_val` 中 label=0 的失败试验 | `results[results["label"] == 0]` |
| 用 `trial_id` 匹配 Part 1 的 `ID` | `.join(reason_lookup, on="trial_id")` |
| 没匹配上的作为单独的 "no recorded stopping reason" 组 | `f[f.source.isna()]` |
| 一个 trial 可以计入多个 reason | `{r: f[f[r] == 1] for r in labels}`,各组独立取子集 |
| **每个 phase 分别分析,phase 内不重复计数** | 按 `phase` 切分 + **`assert f.trial_id.is_unique`** 显式验证 |
| **必须说明哪些是真标签、哪些是预测** | 表中单列 `n true / predicted reason`,逐组给出两个数字 |
| 报告每组的 trial 数和被错判为成功的比例 | `n trials` 和 `frac predicted successful` 两列 |

**`frac predicted successful` 就是该组的假阳性率**(该组全部是真实失败的试验,被预测为成功即是错)。

## 5. 改进措施与 trade-off 评估(2.4 节)⭐ 题目要求"评估改进措施及其 trade-off"

三条独立的 trade-off 分析:

1. **决策阈值扫描** thr ∈ {0.4, 0.5, 0.6, 0.7},每个 phase 报告 **F1 / FPR(failed→success)/ TPR**。
   明确指出 **log loss 和 AUC 不受阈值影响**,阈值只在 F1 和 FPR 之间做交换。
2. **`class_weight="balanced"`** 的 pooled LR,与无加权版对比(图 `p2_err_balanced.png`)。
3. **概率向 phase base rate 收缩**:`p' = π + s·(p − π)`,s ∈ {1.0, 0.9, 0.8, 0.7},在**随机 val 划分**和**时间划分**两套评估上分别算 mean log loss。这是针对 log loss 的过度自信防护。

## 6. ⭐ 时间稳健性检查(2.4 节末)—— 题目明确要求讨论 val→test 差距

> 题目:*"Expect lower performance on the test set than on the validation set, since the test trials are from later years; discuss this gap in your report."*

notebook 不是只是"讨论",而是**构造了一个代理实验去量化它**:

```python
pool = pd.concat([outcome_train, outcome_val], ignore_index=True)
early, late = pool[pool.start_year <= 2008], pool[pool.start_year >= 2010]
# 在 early 上训练,在 late 上评估
```

- **只用已有数据就模拟了 Kaggle 的时间漂移**,不需要 test 标签。
- 留出 2009 年作为缓冲,让两个集合在时间上真正分开。
- 同时跑 baseline 和改进模型,**比较两者在漂移下各损失多少** —— 这比只报告一个数字有信息量得多。

⚠️ **notebook 文字写的是 "test trials start in 2014–2021",而题目说的是 2013–2022。报告里要用题目的 2013–2022。** 详见 `09_issues_to_address.md`。

## 7. 最终预测(2.5 节)

```python
best_s = float(shrink_tab["temporal"].idxmin())            # = 1.0,即不收缩
final_outcome_model = PooledPhaseLR(C=1).fit(pool)         # pool = train ∪ val, 9,050 行
p_test = final_outcome_model.predict_proba(outcome_test)
pi_test = outcome_test.phase.map(pool.groupby("phase").label.mean()).values
p_test = np.clip(pi_test + best_s * (p_test - pi_test), 0.02, 0.98)
```

- 在 **train ∪ val(9,050 行)** 上重训。
- **收缩因子由时间划分实验选出,结果是 1.0(不收缩)** —— 即数据告诉我们模型已经校准得不错。**这是"做了但没采用"的改进,题目说"即使改动没有提升也要讨论",所以必须写进报告。**
- **概率 clip 到 [0.02, 0.98]**:log loss 对接近 0/1 的错误预测惩罚极重(`-ln(0.02) ≈ 3.9`,而 `-ln(0)=∞`),clip 用极小的平均损失换掉尾部灾难风险。这是针对评分指标的直接防护,**理由要在报告里说清楚**。
