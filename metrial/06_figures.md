# 图清单与选图建议

**题目要求:"if you include graphs, be sure to include the source code that generated them"** ⇒ 图的生成代码必须随 `.ipynb` 一起提交到 `midterm-code`。notebook 里每张图都有对应 cell,这一条自动满足,**但报告正文里引用图时最好注明来自哪个 cell**。

✅ **8 张图已全部从 ipynb 中提取,存放在 `metrial/fig/`**(文件名与 notebook 里 `FIG = "figures/"` 写出的一致)。已逐张与 `03`/`05` 里记录的数字核对,**全部一致**;核对发现的三个小问题见本文末 §「核对结果」。
⚠️ **`data/` 仍然缺失**,打包提交 `midterm-code` 时需要补上(见 `09_issues_to_address.md` §A1)。

---

## 全部 8 张图

| 文件名 | 生成 cell | 内容 |
|---|---|---|
| `p1_data.png` | CELL 5 | 左:四个划分(train labeled / train unlabeled / val / test)按 start year 的堆叠直方图。右:五个 reason 在 labeled(157)与 val(78)上的 prevalence 柱状图 |
| `p1_kmeans_diag.png` | CELL 8 | 三联图:K vs inertia / K vs silhouette / K vs label purity(第三幅带 majority baseline 0.583 的虚线) |
| `p1_pca2.png` | CELL 12 | embedding 的 PCA-2 散点(先 L2 归一化),灰点为 unlabeled,彩色点按单标签 trial 的 reason 着色 |
| `p1_ssgmm.png` | CELL 22 | 左:半监督 GMM 每次 EM 迭代的无标签对数似然(70 次迭代的收敛曲线)。右:无标签 pool 最大后验的直方图,红虚线标 0.95 |
| `p1_methods.png` | CELL 23 | 九个方法家族最优配置的横向条形图,四个指标(val/CV × sample/macro F1) |
| `p2_err_baseline.png` | CELL 35 | 三联图(phase I/II/III):baseline 在各 reason 组上的假阳性率,柱子上方标注 n |
| `p2_err_compare.png` | CELL 48 | 同上,**baseline vs improved** 双柱对比 |
| `p2_err_balanced.png` | CELL 50 | 同上,**baseline vs improved vs improved+balanced** 三柱对比 |

### 各图的技术细节

**`p1_data.png`**(CELL 5)
```python
bins = np.arange(1992.5, 2017.5)      # 1993–2016,逐年
ax[0].hist([...4 个划分...], bins=bins, stacked=True)
```
⇒ 数据覆盖 **1993–2016**。右图的数据就是 `01_dataset_facts.md` 的 prevalence 表。

**`p1_kmeans_diag.png`**(CELL 8)
三个子图共用 `km_diag` 表(见 `03_part1_results.md` §2)。第三幅的灰色虚线是 `pd.Series(lab_sets).value_counts().iloc[0] / len(lab_sets) = 0.583`,**即全预测 Recruitment 的 purity**。
⚠️ **ARI 这一列有数据但没有画进图里**(只画了 inertia / silhouette / purity)。**ARI 恰恰是最有说服力的那一列**(K=5 峰值 0.2032 后单调下降)。如果要重做图,**把 ARI 加成第四幅会让"聚类结构弱"的论证更完整**;不重做的话,在正文里用文字给出 ARI 的数字。

**`p1_pca2.png`**(CELL 12)
```python
Z2 = PCA(2, random_state=42).fit_transform(normalize(X_all))
```
只画**单标签** trial 的颜色(`lab_sets == l`,多标签 trial 的 label-set 是 `"Safety|Other"` 这类字符串,不会匹配任何单个 `l`,所以它们落在灰色里)。
notebook 对这张图的文字解读(CELL 13):*"Recruitment trials form one large dense region, while Safety and Efficacy partly overlap ('lack of efficacy' and 'safety concerns' read alike)."*

⚠️ **看过图之后:这个解读不准确,报告里不要照抄。** 图上实际呈现的是:
- **Recruitment(红)是最分散的一类,不是"一大团"** —— 它铺满整个右半侧和上方,同时也混进左侧的密集区。
- **Safety / Funding / Efficacy / Other 四个少数类全部挤在左—中部同一个区域里,互相重叠**(不只是 Safety 与 Efficacy 两类重叠)。
- 所以真实结构是:**一个"四个少数类互相纠缠"的中央区 + 一片几乎纯 Recruitment 的弥散右区。**

⇒ **修正后的解读其实对我们的论证更有利:** 聚类能分出来的只有"Recruitment vs 其余"这一刀(这正好对应 K=2 时 purity 0.5830 恰等于 majority baseline),而**四个少数类在空间上根本不可分**,所以 B0 的 Funding F1 = 0.000、D 的 Funding F1 只有 0.138–0.333 是结构上的必然。建议报告里按这个版本写(`07_discussion_insights.md` 论点 1 已同步修正)。

**`p1_ssgmm.png`**(CELL 22)
就是 Methods 要求的收敛证据(70 次迭代、91.1% 后验 ≥ 0.95)。

**`p1_methods.png`**(CELL 23)
`ax.set_xlim(0.3, 0.95)`,`invert_yaxis()`。数据即 `03_part1_results.md` §4 的 9 行表。
**注意:条形图和表格是同一份数据。5 页限制下两者只放一个。**

**`p2_err_*.png`**(CELL 35 / 48 / 50,共用 `plot_err` 函数)
```python
order = labels + ["No recorded stopping reason", "All failed trials"]
ax.set_ylim(0, 1.12)
# 柱子上方 y=1.02 处标注 n=<trial 数>
```
⚠️ **phase III 的 Funding 组 n=0 ⇒ 该位置是空柱子 + "n=0" 标注。图注里必须说明**,否则看起来像画图出错。

---

## ⭐ 5 页限制下的选图建议

5 页含图,五个章节 + 两个 part 的内容非常挤。**建议最多放 4 张图,优先级如下:**

### 必放(2 张)

1. **`p1_kmeans_diag.png`** —— 支撑整个 Part 1 的方法论("先诊断再设计")。这是 Insight 分数的核心证据:我们**证明了**聚类结构弱,才解释了后面的结果。
   - 如果空间极度紧张,可以只保留 silhouette + purity 两幅(裁掉 inertia,因为"没有 elbow"这个结论用一句话就能说完)。
2. **`p2_err_compare.png`** —— **题目明确要求做 reason 组的错误分析并对比改进前后**。这张图直接对应那条要求,而且一张图承载了 3 个 phase × 7 个组 × 2 个模型 = 42 个数字,信息密度最高。图上自带 n 标注,也同时回答了"考虑类别规模"这一要求。

### 强烈建议(1 张,二选一)

3a. **`p1_ssgmm.png`** —— **Methods 章节被点名要求写 EM 的收敛阈值和迭代次数**。这张图是最直接的证据,而且右半幅(91% 的后验 ≥ 0.95)支撑了 Discussion 里"自信但错误"的关键论点。
3b. **`p1_pca2.png`** —— 定性上最直观地显示了"四个少数类挤在同一区域互相纠缠,Recruitment 弥散在外围"。

> **推荐选 3a**,因为它对应一条**硬性的评分要求**,而 3b 的内容可以用一句文字说清。

### 可选(1 张)

4. **`p1_methods.png`** 或 **§4 的 9 行表格**(二选一,不要都放)。
   **建议放表格而不是条形图** —— Results 章节**明确要求"a table summarizing the performance of the models"**,表格直接满足要求,而且 4 个指标 × 9 个方法用表格读起来比条形图更精确。省下的空间给 Part 2 的 baseline-vs-improved 表。

### 建议不放

- **`p1_data.png`** —— 年份分布和 prevalence 都可以用 1–2 句话 + `01_dataset_facts.md` 里的小表交代。"有标签 trial 集中在 2003 年前,test 是 2010–2016"一句话就够。
- **`p2_err_baseline.png`** —— 完全被 `p2_err_compare.png` 包含(后者的第一组柱子就是它)。
- **`p2_err_balanced.png`** —— `class_weight="balanced"` 是一个"尝试了但无效"的改动,**用 §7 的两行表格(mean AUC 0.7612→0.7609, LL 0.5665→0.5685)+ 一句解释就足够**,不值得占一张图的版面。

### 版面预算参考

| 章节 | 建议页数 | 内容 |
|---|---|---|
| 封面 | 不计入正文页数(但要确认模板怎么算) | 课程名、全部组员姓名、**每人负责的工作** |
| Abstract | 0.2 页 | |
| Methods | 1.5 页 | Part 1 的 A–H + ss_gmm 的初始化/收敛细节;Part 2 的 field-wise TF-IDF + phase augmentation |
| Results | 1.5 页 | **Part 1 的 9 行汇总表 + Part 2 的 baseline/improved 表**,2–3 张图 |
| Discussion | 1.3 页 | Part 1 四条 takeaway + Part 2 错误分析解读 + trade-off + 时间漂移 |
| References | 第 5–6 页,不计入 | 见 `08_references.md` |

---

## 核对结果(逐张看过 `fig/` 里的 8 张 PNG)

**结论:8 张图全部存在,图上的数值与 `03`/`05` 里记录的表格完全一致**,没有发现数字性错误。逐张确认:

| 图 | 核对内容 | 结果 |
|---|---|---|
| `p1_data.png` | 有标签 train 集中在 2003 年前(另有 2004–2009 的零散蓝柱 = 为稀有原因额外挑选的 40 个);test 落在 2010–2016;右图 prevalence 与 `01` 的表一致 | ✅ |
| `p1_kmeans_diag.png` | inertia 1101.96 → 389.01 平滑下降无 elbow;silhouette K=2 的 0.1964 → K=3 的 0.1628 → 单调升至 0.3070;purity 0.583 → 0.8723,灰虚线在 0.583 | ✅ |
| `p1_pca2.png` | 见上文 ⚠️,**图本身没问题,是 notebook 的文字解读不准** | ⚠️ 解读需修正 |
| `p1_ssgmm.png` | 左:对数似然单调上升并在约 70 次迭代处趋平;右:直方图最高柱在 0.95 右侧,红虚线在 0.95 | ✅ |
| `p1_methods.png` | 9 个家族 × 4 个指标,与 §4 的表一致 | ✅(有小瑕疵,见下) |
| `p2_err_baseline.png` | 三个 phase 的假阳性率与 §4 的 21 行表一致 | ✅ |
| `p2_err_compare.png` | baseline vs improved 双柱,与 §5 的表一致;phase I Funding 的 improved 柱为 0、phase III Funding 两柱皆缺 | ✅(有小瑕疵,见下) |
| `p2_err_balanced.png` | 三柱,balanced 在 phase II 各组普遍更低、phase III Safety 降到 0.429 | ✅(有小瑕疵,见下) |

### 三个排版小瑕疵(不影响数字,但放进报告前值得知道)

1. **⭐ `p2_err_compare.png` 和 `p2_err_balanced.png` 的 Phase III 子图里,图例挡住了右侧两组的 `n=` 标注**(`n=82` 和 `n=123` 被压在图例框下面,呈灰色半透明)。
   - **影响:** 题目点名要求"考虑类别规模",而最大的两组恰好是被挡住的那两个。
   - **处理:** 最省事的办法是**在图注里把 n 写出来**("No reason: n = 28 / 144 / 82; All failed: n = 64 / 265 / 123")。如果能重跑,把 `legend` 移到 `loc="upper left"` 或 `bbox_to_anchor` 到图外即可。
2. **`p1_methods.png` 的图例盖住了最下面一行(G label spread)的部分条形。** G 本身是我们最差的方法,遮挡不影响主要信息;而且按 §「选图建议」这张图建议**用 §4 的表格替代**,所以可以不管。
3. **`p1_kmeans_diag.png` 仍然没有 ARI 子图**(只有 inertia / silhouette / purity)。ARI 是支撑"聚类结构弱"最有力的一列,**在正文里用一句话给出数字即可**:"ARI against the true label-sets peaks at only 0.2032 (K = 5) and declines for larger K."

### 另外一个值得写进报告的观察(来自看图,表里看不出来)

`p1_kmeans_diag.png` 的 silhouette 曲线**在 K=3 处降到最低(0.1628),然后随 K 单调上升到 K=80 的 0.3070 且没有转折**。
⇒ 这比"silhouette 很低"更有说服力:**silhouette 随 K 单调上升而不收敛,是"簇在不断碎片化"的典型特征**(把一个连续分布切得越碎,每个碎片内部当然越紧致)。配合 ARI 在同一区间**单调下降**(K=5 的 0.2032 → K=80 的 0.0760),两条曲线反向移动,**共同排除了"存在某个真实 K"的可能性**。这一句是整个 §1.4 诊断最强的表述,建议写进报告。
