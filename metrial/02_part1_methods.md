# Part 1 方法细节(Methods 章节素材)

> 评分细则点名要求写:**聚类质心 / GMM 分量如何初始化**、**EM 的收敛阈值和迭代次数**。本文件把这些都标成 ⭐。

## 0. 公共预处理

```python
labels   = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
features = [f"emb_{i}" for i in range(150)]
labeled  = (reason_train[labels] != -100).all(axis=1).values   # 157 个 True

scaler  = StandardScaler()
X_train = scaler.fit_transform(reason_train[features])   # fit 用全部 1171 行(labeled+unlabeled)
X_val   = scaler.transform(reason_val[features])
X_test  = scaler.transform(reason_test[features])
```

- **标准化的 fit 只用 `part1_train.csv`(含被遮蔽的行),不含 val 和 test。** 这是一个诚实的做法,报告里可以提一句。
- 记号(全文沿用):
  - `XL, YL` = 157 个有标签 train 样本(157×150, 157×5)
  - `XU` = **unlabeled pool,1,326 行 = 1,014 个遮蔽的 train + 312 个 test**
  - `Yva` = val 标签(78×5)
  - `XP, YP` = `vstack([XL, X_val])`, `vstack([YL, Yva])` = **235 个有标签样本**,用于 CV 和最终重训

### ⭐ 转导设定(必须在报告中说明)

`XU` **包含 test 的特征(但不含任何 test 标签)**,所以所有半监督方法都是**转导式(transductive)**的。

- 这是合法的半监督学习,**不是题目禁止的"反推 test 标签"** —— 我们只用了 test 的 `emb_*` 特征,从未使用任何 test 标签信息,也没有去外部找数据源。
- **建议在报告里主动写明这一点**,否则容易被误解。
- 一个由此产生的不对称性(也应说明):`XU` **不含 val**,CV 的留出折也不在 pool 里。因此 **val/CV 分数是归纳式(inductive)的,而 test 预测是转导式的** —— 实际 test 表现可能略优于验证估计。

## 1. 两个关键后处理 / 工具函数

### `at_least_one` —— 多标签解码规则

```python
def at_least_one(P, thr=0.5):
    Y = (P >= thr).astype(int)
    empty = Y.sum(1) == 0
    Y[empty, P[empty].argmax(1)] = 1     # 没有标签过阈值时,取概率最大的那个
    return Y
```

**论证(报告里值得单独写一句):** Part 1 的每个 trial 都是**因为某个原因而停止的**(数据集定义就是"有记录的停止原因"),所以空预测在逻辑上不可能正确,而且在 **sample-wise F1 下空预测直接得 0 分**。这个后处理把一个纯粹的度量损失修掉了,并且与"几乎每个 trial 恰好一个 reason"(147/157)的观察一致。

### `fit_lr` —— 五个独立的二分类 LR(即 one-vs-rest)

```python
def fit_lr(X, Y, C=0.1, w=None, class_weight="balanced"):
    models = [LogisticRegression(C=C, max_iter=3000, class_weight=class_weight)
                .fit(X, Y[:, j], sample_weight=w)
              for j in range(Y.shape[1])]
    return lambda Q: np.column_stack([m.predict_proba(Q)[:, 1] for m in models])
```

- L2 正则(sklearn 默认),默认 solver(lbfgs),`max_iter=3000`。
- **`class_weight="balanced"`** 的动机:Funding 只有 8 个正例。
- `sample_weight=w` 用于**给伪标签降权**(`w_u`)。
- ⚠️ **注意**:sklearn 会把 `class_weight` 和 `sample_weight` **相乘**,所以 `w_u=0.3` 的真实含义是"在类别再平衡之后再乘 0.3",不是绝对权重 0.3。如果报告里要解释 `w_u`,用这个措辞更准确。

## 2. 评估协议 ⭐(Methods 必写)

```python
def evaluate(name, method, cv=True, **kw):
    yv = at_least_one(method(XL, YL, XU, **kw)(X_val))        # 验证集分数
    ...
    for rep in range(2):                                       # 2 次重复
        for a, b in KFold(5, shuffle=True, random_state=rep).split(XP):   # 5 折
            yh = at_least_one(method(XP[a], YP[a], XU, **kw)(XP[b]))
```

两套指标同时报告,理由明确:

1. **验证集分数**(78 个 trial)—— 这是题目指定的选模集。
2. **2×5 折交叉验证,在 235 个有标签样本(train-labeled ∪ val)上** —— 因为**验证集只有 78 个 trial,一个 trial ≈ 1.3 个百分点的 sample-F1**,单看 val 噪声太大。
   - 每折中留出折的标签被隐藏,**每个方法看到的 unlabeled pool 完全相同**(都是同一个 `XU`),保证可比性。
   - 共 10 次拟合,取均值。

**这是一个很强的 Insight 得分点:** 明确量化"一个 trial = 1.3 分"并据此引入 CV,说明我们知道验证集的统计功效有限,而不是盲目追 val 分数。

## 3. Baseline(starter code)

```python
reason_model = OneVsRestClassifier(
    LogisticRegression(max_iter=500, solver="liblinear", random_state=42))
reason_model.fit(X_train[labeled], y_train)
val_pred = reason_model.predict(X_val)
```

- 即 **C=1(默认)、liblinear、无 `class_weight`、无 `at_least_one` 后处理**,只用 157 个有标签样本。
- 结果见 `03_part1_results.md`。

## 4. 八个 label-recovery 方法(A–H)

原 notebook 的方法表(可直接用于报告):

| id | 方法 | 无监督部分 |
|---|---|---|
| A | 157 个有标签样本上的监督 one-vs-rest LR | 无(参照基准) |
| B | **K-means cluster-then-label**:每个簇的标签向量 = 其有标签成员的平滑均值;unlabeled 样本取所属簇的标签;LR 在 labeled + 伪标签(权重 `w_u`)上重训 | K-means(k-means++,10 次重启) |
| B0 | **纯聚类分类器**:query 样本直接取其所属 K-means 簇的标签分布(不接 LR) | K-means |
| C | **K-means ∩ LR 一致性**:只对"簇投票"与"监督 LR"**一致**的 unlabeled 样本打伪标签 | K-means |
| D | **GMM 软标签**:PCA-20 上的对角协方差 GMM;unlabeled 样本从分量后验得到软标签;LR 用分数权重训练 | GMM(EM) |
| E0 | **半监督 GMM(Nigam 风格 EM)分类器**:每个在有标签数据中出现过的 label-set 一个高斯,**tied(共享)协方差**;labeled 的 responsibility 被钉住,unlabeled 的由 E 步估计 | GMM + label-seeded EM |
| E | E0 的后验 → 置信伪标签 → LR | 同上 |
| F | **Self-training LR**:反复把五个概率都足够置信的 unlabeled 样本加入训练集 | 无(wrapper) |
| G | **Label spreading**(kNN 图,每个标签跑一次) | 图方法 |
| H | **Ensemble**:A、C、E 三者的概率平均 | — |

### ⭐ A — 监督 LR(参照)

```python
def m_supervised(XL, YL, XU, C=0.1):
    return fit_lr(XL, YL, C)
```
扫过 **C ∈ {0.01, 0.1, 1, 10}**。不使用 `XU`。

### ⭐ B / B0 — K-means cluster-then-label

```python
def kmeans_cluster_labels(XL, YL, XU, K, npc, seed=42):
    Z = PCA(npc, random_state=seed).fit_transform(normalize(np.vstack([XL, XU])))
    c = KMeans(K, n_init=10, random_state=seed).fit_predict(Z)
    cl, cu = c[:len(XL)], c[len(XL):]
    prior = YL.mean(0)
    CL = np.array([(YL[cl == k].sum(0) + prior) / ((cl == k).sum() + 1) for k in range(K)])
    return CL[cu]
```

**初始化与设定(⭐ 评分细则点名):**
- **质心初始化:k-means++,`n_init=10`(10 次独立重启,取 inertia 最小的一次)**,`random_state=42`。
- 预处理:**先 L2 归一化**(`normalize`)**再 PCA 到 `npc` 维**。L2 归一化使欧氏距离与余弦距离单调相关,这对 sentence embedding 是惯例做法。
- **PCA 的 fit 在 `vstack([XL, XU])` 上,不使用标签** —— 聚类部分是纯无监督的。
- 默认 `npc=50`。
- **簇标签的平滑方式**:`(该簇有标签成员的标签和 + 全局先验) / (该簇有标签成员数 + 1)`,即**向有标签集的全局先验做 add-one 平滑**。动机:很多簇里有标签成员极少甚至为零(157 个标签散在 K 个簇里),不平滑会得到退化的 0/1 或除零。
- B 的超参扫描:**K ∈ {10, 20, 40, 80} × w_u ∈ {0.3, 1.0}**;B0:**K ∈ {10, 20, 40, 80}**。

B 的 LR 重训:
```python
def m_kmeans(XL, YL, XU, K=40, npc=50, C=0.1, wu=0.3):
    YU = at_least_one(kmeans_cluster_labels(XL, YL, XU, K, npc))
    return fit_lr(np.vstack([XL, XU]), np.vstack([YL, YU]), C,
                  w=np.r_[np.ones(len(XL)), wu * np.ones(len(XU))])
```
B0(纯聚类,无 LR):query 与 `XU` 拼在一起重新聚类,然后取尾部对应行。

### ⭐ C — K-means ∩ LR 一致性过滤

```python
def m_kmeans_agree(XL, YL, XU, K=20, npc=50, C=0.1, wu=1.0):
    Yc = at_least_one(kmeans_cluster_labels(XL, YL, XU, K, npc))   # 簇投票
    Ys = at_least_one(fit_lr(XL, YL, C)(XU))                       # 监督 LR 预测
    keep = (Yc == Ys).all(1)                                       # 五个标签全部一致
    return fit_lr(np.vstack([XL, XU[keep]]), np.vstack([YL, Ys[keep]]), C,
                  w=np.r_[np.ones(len(XL)), wu * np.ones(keep.sum())])
```

- 保留条件是**五个标签位全部相同**(`.all(1)`),是个很严的过滤。
- 被保留样本用的是 **LR 的预测 `Ys`** 作为伪标签(不是簇投票 `Yc`)。
- 伪标签权重 `w_u=1.0`(不降权),因为已经过滤过了。
- 扫描 **K ∈ {10, 20, 40}**。
- **这是 co-training / 多视角一致性的思想**:聚类(生成式、无监督视角)与 LR(判别式、有监督视角)有不同的错误模式,两者一致的样本更可信。

### ⭐ D — GMM 软标签

```python
def m_gmm_soft(XL, YL, XU, K=20, npc=20, C=0.1, wu=0.5, seed=42):
    Z = PCA(npc, random_state=seed).fit_transform(np.vstack([XL, XU]))
    g = GaussianMixture(K, covariance_type="diag", reg_covar=1e-3,
                        n_init=3, random_state=seed).fit(Z)
    R = g.predict_proba(Z); Rl, Ru = R[:len(XL)], R[len(XL):]
    prior = YL.mean(0)
    PU = Ru @ ((Rl.T @ YL + prior) / (Rl.sum(0)[:, None] + 1))
    ...
```

**初始化与设定(⭐):**
- `sklearn.mixture.GaussianMixture`,**`covariance_type="diag"`(对角协方差)**,**`reg_covar=1e-3`**(协方差对角加正则,防奇异),**`n_init=3`**(3 次重启)。
- **分量初始化:sklearn 默认 `init_params="kmeans"`**,即用 K-means 的结果初始化分量的均值/权重/协方差。**这一点要在报告里写明**,因为评分细则专门问 GMM 如何初始化。
- 这里 **PCA 之前不做 L2 归一化**(与 B/C 不同),因为 GMM 假设的是欧氏空间里的高斯,归一化到球面会破坏这个假设。`npc=20`。
- **软标签构造**:`PU = Ru @ (分量的标签均值)`,其中分量标签均值同样用 add-one 向全局先验平滑。即"先算每个高斯分量的平均标签,再按后验加权给 unlabeled 样本"。
- **软标签如何喂给 LR**(实现细节,值得写):把每个 unlabeled 样本**复制成两行**,一行 y=1 权重 `w_u·PU`,一行 y=0 权重 `w_u·(1-PU)`。这样用分数权重精确表达了软标签,而不是硬化成 0/1。
- `w_u=0.5`。扫描 **K ∈ {10, 20, 40}**。

### ⭐⭐ E0 / E — 半监督 GMM(Nigam 风格 EM),这是 Methods 的重点

完整实现与逐项说明:

```python
def ss_gmm(XL, YL, XU, npc=20, lam=1.0, reg=1e-2, max_iter=100, tol=1e-5):
    pca = PCA(npc, random_state=42).fit(np.vstack([XL, XU]))
    ZL, ZU = pca.transform(XL), pca.transform(XU)
    combos, cls = np.unique(YL, axis=0, return_inverse=True)
    K = len(combos); RL = np.eye(K)[cls.ravel()]; Z = np.vstack([ZL, ZU])

    def mstep(RU):
        R = np.vstack([RL, lam * RU]); Nk = R.sum(0) + 1e-6
        pi, mu = Nk / Nk.sum(), (R.T @ Z) / Nk[:, None]
        S = sum((R[:, k:k+1] * (Z - mu[k])).T @ (Z - mu[k]) for k in range(K)) / Nk.sum() \
            + reg * np.eye(npc)                      # tied covariance + ridge
        return pi, mu, np.linalg.inv(S), np.linalg.slogdet(S)[1]

    def logp(Zq, pi, mu, Si, ld):
        return np.column_stack([np.log(pi[k]) - 0.5 * (
            np.einsum("ij,jk,ik->i", Zq - mu[k], Si, Zq - mu[k]) + ld) for k in range(K)])

    params = mstep(np.zeros((len(ZU), K)))           # ⭐ 初始化:只用有标签数据
    hist = []
    for it in range(max_iter):
        lp = logp(ZU, *params); ll = logsumexp(lp, 1).sum(); hist.append(ll)
        RU = np.exp(lp - logsumexp(lp, 1, keepdims=True))        # E 步
        params = mstep(RU)                                       # M 步
        if it > 0 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2]):   # ⭐ 收敛判据
            break

    def predict(Q):
        lp = logp(pca.transform(Q), *params)
        return np.exp(lp - logsumexp(lp, 1, keepdims=True)) @ combos   # 后验 → 标签概率
    return predict, RU, combos, hist
```

**模型结构:**
- **分量 = 有标签数据中出现过的每一个 label-set**(`np.unique(YL, axis=0)`)。
  - 由于 labeled 集里 147 个单标签 + 10 个双标签,分量数 K = 5 个单 reason + 实际观测到的若干组合。
  - ⚠️ **K 的确切数值需要跑一次数据才能确认**(见 `09_issues_to_address.md`),报告里要给具体数字。
- **tied(共享)协方差**:所有分量共用一个协方差矩阵 `S`,加 **ridge 正则 `reg=1e-2`**(`+ reg·I`)。动机:npc=20 维、只有 157 个有标签样本,每个分量各自估计一个 20×20 协方差会严重过拟合/奇异。
- 在 **PCA-`npc` 空间**上拟合,PCA 的 fit 在 `vstack([XL, XU])` 上。扫 `npc ∈ {10, 20, 40}`。

**⭐ 初始化(评分细则点名):**
- `params = mstep(np.zeros((len(ZU), K)))` —— **第一次 M 步时所有 unlabeled 的 responsibility 都设为 0,即仅用有标签数据初始化分量均值、先验和共享协方差。**
- labeled 样本的 responsibility 矩阵 `RL = np.eye(K)[cls]` 是**one-hot 硬钉死的**,在所有迭代中保持不变(不参与 E 步更新)。这正是 Nigam et al. (2000) 的 labeled-data-clamped EM。

**⭐ 训练 / 收敛(评分细则点名):**
- E 步:只对 unlabeled 样本计算后验 responsibility `RU`(用 logsumexp 做数值稳定的归一化)。
- M 步:`R = vstack([RL, lam·RU])`,即**unlabeled 的 responsibility 以权重 `λ` 进入 M 步**。`λ` 控制 unlabeled 数据的影响力,扫 **λ ∈ {0.0, 0.3, 1.0}**(λ=0 退化为纯监督的生成式分类器)。
- **收敛判据:unlabeled 的对数似然的相对变化 < `tol = 1e-5`**,即 `|ll_t - ll_{t-1}| < 1e-5 · |ll_{t-1}|`。
- **`max_iter = 100`**(上限)。
- **实测:npc=20, λ=1.0 时在 70 次迭代后收敛**;收敛曲线见图 `p1_ssgmm.png`。
- **实测:收敛后 unlabeled pool 中 91.1% 的样本最大后验 ≥ 0.95**。

**预测:** query 的分量后验 @ `combos` 矩阵 ⇒ 每个 reason 的边缘概率。这是一个**生成式多标签分类器**(E0)。

**E(接 LR):**
```python
def m_ss_gmm_lr(XL, YL, XU, npc=20, lam=1.0, conf=0.95, wu=0.3, C=0.1):
    _, RU, combos, _ = ss_gmm(XL, YL, XU, npc=npc, lam=lam)
    keep = RU.max(1) >= conf
    return fit_lr(np.vstack([XL, XU[keep]]), np.vstack([YL, combos[RU.argmax(1)][keep]]),
                  C, w=np.r_[np.ones(len(XL)), wu * np.ones(keep.sum())])
```
- **置信度阈值 `conf`**:只保留最大后验 ≥ conf 的 unlabeled 样本,伪标签取其 MAP label-set。扫 **conf ∈ {0.8, 0.95}**、**npc ∈ {20, 40}**,`w_u=0.3`。

### ⭐ F — Self-training LR

```python
def m_self_train(XL, YL, XU, C=0.1, thr=0.95, rounds=5):
    X, Y, rem = XL, YL, np.arange(len(XU))
    for _ in range(rounds):
        P = fit_lr(X, Y, C)(XU[rem])
        conf = (np.abs(P - 0.5) * 2).min(1) >= thr     # 五个概率都要够极端
        if not conf.any(): break
        X, Y, rem = np.vstack([X, XU[rem[conf]]]), np.vstack([Y, at_least_one(P[conf])]), rem[~conf]
    return fit_lr(X, Y, C)
```
- **置信度定义:`min_j |2·p_j - 1| ≥ thr`**,即**五个标签的概率都必须离 0.5 足够远**(全部确定,不只是最大那个)。
- **最多 5 轮**,每轮把通过的样本移出候选池(`rem[~conf]`),**伪标签权重为 1(不降权)**。
- 扫 **thr ∈ {0.8, 0.95}**。

### ⭐ G — Label spreading

```python
def m_label_spreading(XL, YL, XU, k=20, alpha=0.5):
    X = normalize(np.vstack([XL, XU]))
    models = [LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha, max_iter=200)
                .fit(X, np.r_[YL[:, j], -np.ones(len(XU), int)]) for j in range(5)]
```
- `sklearn.semi_supervised.LabelSpreading`,**kNN 图核**,`alpha=0.5`(clamping factor),**`max_iter=200`**。
- **在 L2 归一化后的原始 150 维空间**上建图(不降维)。
- **每个标签独立跑一次**(五个二分类问题),unlabeled 用 `-1` 标记。
- 扫 **k ∈ {10, 20}**。

### ⭐ H — Ensemble(最终选定)

```python
def m_ensemble(XL, YL, XU):
    fs = [m_supervised(XL, YL, XU),      # A, C=0.1
          m_kmeans_agree(XL, YL, XU),    # C, K=20, npc=50, C=0.1, wu=1.0
          m_ss_gmm_lr(XL, YL, XU)]       # E, npc=20, lam=1.0, conf=0.95, wu=0.3
    return lambda Q: np.mean([f(Q) for f in fs], 0)
```
- **等权重**的概率平均(不是投票),之后再过 `at_least_one`。
- 选这三个的理由:**三者的错误模式不同** —— A 纯判别式、C 用聚类一致性过滤的伪标签、E 用生成式模型的伪标签。

## 5. 特征选择实验(1.6 节)

在固定配置(**监督 LR 部分,C=0.1**)上逐组加特征,只看 val:

| 特征组 | 构造方式 |
|---|---|
| embeddings only | 150 维标准化 embedding |
| + start_year | `(start_year - 2005) / 3`,1 维 |
| + criteria/disease/drug text | `criteria + diseases_cleaned + drugs` 拼接 → TF-IDF(`min_df=3, max_features=20000, stop_words="english", sublinear_tf=True`)→ **TruncatedSVD 50 维** → StandardScaler。TF-IDF 和 SVD 的 fit 在 `concat([reason_train, reason_test])` 上 |
| + drug match flag + drug actions | `matched_drugs.notna()` 标志位(1 维)+ `drug_actions` 的 MultiLabelBinarizer(按 `;` 切分) |
| + level-0 Reactome pathways | `drug_pathways` 的 MultiLabelBinarizer,**只保留 `pathway_catalog` 中 `pathway_level == 0` 的 29 个顶层通路** |
| + everything | 上面四组全加 |

**设计说明(可写进报告):** 只取 level-0 通路是有意的降维选择 —— Reactome 是层级结构,深层通路极其稀疏且与顶层高度共线(属于具体通路的靶点也被列在所有更宽泛的通路下),用 29 个顶层类别能在保留生物学信号的同时控制维度。

## 6. 最终模型与产出(1.7 节)

```python
# 验证集上的最终评估:只用 157 个有标签样本训练
final_val = at_least_one(m_ensemble(XL, YL, XU)(X_val))

# test 预测 & 恢复遮蔽标签:在 235 个有标签样本(train+val)上重训,unlabeled pool 不变
final_reason_model = m_ensemble(XP, YP, XU)
Y_test1  = at_least_one(final_reason_model(X_test))
Y_unlab  = at_least_one(final_reason_model(X_train[~labeled]))
```

- **选模用 val,选定后在 train∪val 上重训** —— 这是标准做法,报告里一句话交代即可。
- 产出三个文件:
  - `part1_submission.csv` —— Kaggle 提交(312 行,`ID,Predicted`)
  - `part1_train_recovered_labels.csv` —— 增广后的训练标签(1,171 行,带 `label_source` 列标记 `true`/`recovered`)
  - `reason_lookup` —— Part 2 错误分析用的查找表,**1,326 predicted + 235 true**

## 7. 聚类结构诊断(1.4 节,先于方法设计)

```python
X_all = np.vstack([XL, XU, X_val])                          # 1,561 行
Z_all = PCA(50, random_state=42).fit_transform(normalize(X_all))
for K in [2,3,5,8,10,15,20,30,40,60,80]:
    km = KMeans(K, n_init=10, random_state=42).fit(Z_all)
```

三个诊断量:
- **inertia**(找 elbow)
- **silhouette**(在全部 1,561 个样本上算,**不用标签**)
- **label purity**:对 235 个有标签样本,"其 label-set 等于所属簇的多数 label-set"的比例
- **ARI**(簇划分 vs label-set 划分)

**这一节的作用是"先诊断再设计"**,是 Insight 的得分点:我们不是盲目套聚类,而是**先验证了聚类结构很弱,再据此解释为什么纯聚类方法(B0/D/G)会失败、为什么伪标签必须过滤**。
</content>
