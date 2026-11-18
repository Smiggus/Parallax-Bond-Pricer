"""Build the final-revision notebooks: every ML II algorithm applied to the bond pricing project.

Run with the mbds interpreter:
    python build_project_notebooks.py            # write the notebooks
    python build_project_notebooks.py --execute  # write and run them (needs the panel, see rev_utils.py)

Each notebook pairs a course concept with a project question, asks you to predict the result before
running a cell ("Predict first"), and closes with "Concept" notes. Only train and valid are used.
"""

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
PY = sys.executable

SETUP = """\
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from rev_utils import load, sample, baseline_valid, score, table, save, make_preprocessor, xy, SEED
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)
train, valid = load()
X_tr, y_tr = xy(train); X_va, y_va = xy(valid)
valid = valid.loc[y_va.index].reset_index(drop=True); X_va = X_va.reset_index(drop=True); y_va = y_va.reset_index(drop=True)
base = baseline_valid(valid)
rows = [score(valid, base, "matrix pricing (same date)")]
print(f"train {len(X_tr):,} rows, valid {len(X_va):,} rows, {X_tr.shape[1]} features")
table(rows)"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (mbds)", "language": "python"}
    return nb


# ---------------------------------------------------------------- 1. Trees
R1 = [
    md("""# R1 - Decision trees on bond spreads

**Course topic:** Classification and regression trees (splits, impurity, depth, pruning).
**Project question:** can one tree price a bond better than matrix pricing (median spread of the same grade x maturity x sector cell)?

Matrix pricing *is* a hand-built tree: three fixed splits (grade, maturity bucket, sector) and the leaf predicts a median.
A regression tree learns its own splits and predicts the leaf **mean**. That is the whole idea of this notebook."""),
    code(SETUP),
    md("""## 1. Trees do not need scaling
Trees split on one feature at a time with a threshold (`vol_60d <= 0.31`), so the units do not matter.
We use the recipe with `scale=False` and integer codes for categories."""),
    code("""from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text
pre = make_preprocessor(scale=False, one_hot=False)
Xt = pre.fit_transform(X_tr); Xv = pre.transform(X_va); names = pre.get_feature_names_out()
Xt.shape"""),
    md("""## 2. Read a small tree
**Predict first:** which feature will be the root split? (Hint: what separates a 90 bp bond from a 400 bp bond?)"""),
    code("""stump = DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(Xt, y_tr)
print(export_text(stump, feature_names=list(names), decimals=2))
rows.append(score(valid, stump.predict(Xv), "tree depth 3")); table(rows)"""),
    md("""**Concept:** a regression tree's criterion is squared error (variance reduction); each leaf predicts the
**mean** of its training rows. A split is chosen to minimise the weighted sum of child impurities.
Count the leaves: depth 3 means at most 2^3 = 8 leaves, so at most 8 different predicted spreads.

Reading categories in the printout: they are integer codes in alphabetical order, so `grade_bucket <= 0.50` means
HY (code 0) and `> 0.50` means IG (code 1). A tree can split nominal codes, but the order means nothing."""),
    md("""## 3. Depth: underfit vs overfit
**Predict first:** as depth grows, what happens to train MAE? To valid MAE?"""),
    code("""from sklearn.metrics import mean_absolute_error as mae
depths = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, None]
curve = []
for d in depths:
    t = DecisionTreeRegressor(max_depth=d, min_samples_leaf=1, random_state=SEED).fit(Xt, y_tr)
    curve.append({"depth": d or "none", "leaves": t.get_n_leaves(),
                  "train_mae": mae(y_tr, t.predict(Xt)), "valid_mae": mae(y_va, t.predict(Xv))})
curve = pd.DataFrame(curve); print(curve.round(2).to_string(index=False))
ax = curve.plot(x="depth", y=["train_mae", "valid_mae"], marker="o", figsize=(7, 3.5)); ax.set_ylabel("MAE bps")
ax.axhline(rows[0]["mae_bps"], ls="--", c="grey", label="matrix pricing"); ax.legend(); plt.show()"""),
    md("""**Concept:** train error always falls with depth (a full tree memorises the training set: train MAE near 0).
Valid error falls, bottoms out, then rises: that is **overfitting** (high variance). Shallow trees underfit (high bias)."""),
    md("""## 4. Cost-complexity pruning (`ccp_alpha`)
Grow a big tree, then cut back branches whose error reduction is not worth their extra leaves:
minimise `error + alpha x number_of_leaves`. Larger alpha means a smaller tree."""),
    code("""# The path is computed on the full training set with no other limit, so it covers every possible subtree.
path = DecisionTreeRegressor(random_state=SEED).cost_complexity_pruning_path(Xt, y_tr)
# Most of the path's alphas are tiny (they remove one leaf each), so space the candidates on a log scale instead.
alphas = np.r_[0, np.geomspace(0.01, path.ccp_alphas[-2], 20)]
prune = []
for a in alphas:
    t = DecisionTreeRegressor(ccp_alpha=a, random_state=SEED).fit(Xt, y_tr)
    prune.append({"alpha": a, "leaves": t.get_n_leaves(), "valid_mae": mae(y_va, t.predict(Xv))})
prune = pd.DataFrame(prune); best = prune.loc[prune.valid_mae.idxmin()]
print(prune.round(3).to_string(index=False)); print("best:", best.round(3).to_dict())
ax = prune.plot(x="leaves", y="valid_mae", marker="o", logx=True, figsize=(7, 3.5), legend=False)
ax.set_ylabel("valid MAE bps"); ax.set_title("pruning path: fewer leaves as alpha grows"); plt.show()
best_tree = DecisionTreeRegressor(ccp_alpha=best.alpha, random_state=SEED).fit(Xt, y_tr)
rows.append(score(valid, best_tree.predict(Xv), f"tree pruned, alpha {best.alpha:.3g} ({int(best.leaves)} leaves)"))
table(rows)"""),
    md("""**Concept:** pruning trades a little bias for a lot less variance. In class we picked alpha by
cross-validation; here we pick it on the **chronological** valid months, because a random split would leak
(the same bond appears every month, with nearly the same spread).

Two honest caveats: (1) alpha and depth are chosen on the valid months and scored on the same months, so the valid
score is a little optimistic; a clean estimate needs the test months, which stay closed. (2) Matrix pricing uses
other issuers' spreads **on the same valid date**, which the models never see, so it is a desk benchmark, not a
like-for-like model."""),
    md("""## 5. A classification tree: will the spread widen by 25 bp next month?
`widens_25bp` is 1 for about 4% of rows. **Predict first:** what accuracy does "always say no" get?"""),
    code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
Xc_tr, yc_tr = xy(train, "widens_25bp"); Xc_va, yc_va = xy(valid, "widens_25bp")
prec = make_preprocessor(scale=False, one_hot=False); Ct = prec.fit_transform(Xc_tr); Cv = prec.transform(Xc_va)
yc_tr = yc_tr.astype(int); yc_va = yc_va.astype(int)
print("share of 1s - train %.3f, valid %.3f" % (yc_tr.mean(), yc_va.mean()))
print("always-0 accuracy on valid: %.3f" % accuracy_score(yc_va, np.zeros_like(yc_va)))
cls = {}
for cw in [None, "balanced"]:
    c = DecisionTreeClassifier(max_depth=5, criterion="gini", class_weight=cw, random_state=SEED).fit(Ct, yc_tr)
    p = c.predict(Cv)
    cls[str(cw)] = {"accuracy": accuracy_score(yc_va, p), "balanced_acc": balanced_accuracy_score(yc_va, p),
                    "auc": roc_auc_score(yc_va, c.predict_proba(Cv)[:, 1])}
    print(cw, "\\n", confusion_matrix(yc_va, p))
pd.DataFrame(cls).T.round(3)"""),
    md("""**Concept:** Gini of a node = 1 - sum(p_k^2). A node with 4% ones has Gini = 1 - (0.96^2 + 0.04^2) = 0.077,
already "pure-looking", so an unweighted tree barely bothers splitting the rare class: high accuracy, useless model.
`class_weight="balanced"` reweights the rare class. Judge imbalanced problems with recall, balanced accuracy or AUC,
never raw accuracy."""),
    code("""save("r1_trees", {"table": rows, "depth_curve": curve.astype({"depth": str}).to_dict("records"),
                   "best_alpha": float(best.alpha), "best_leaves": int(best.leaves), "classifier": cls,
                   "share_widen_valid": float(yc_va.mean())})
table(rows)"""),
]

# ---------------------------------------------------------------- 2. Ensembles
R2 = [
    md("""# R2 - Bagging, random forest and boosting on bond spreads

**Course topic:** Bootstrap aggregation, random feature subsets, OOB error, feature importance, boosting.
**Project question:** how much do ensembles gain over one pruned tree and over matrix pricing?"""),
    code(SETUP),
    code("""from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
pre = make_preprocessor(scale=False, one_hot=False)
Xt = pre.fit_transform(X_tr); Xv = pre.transform(X_va); names = pre.get_feature_names_out()"""),
    md("""## 1. Bagging: many deep trees on bootstrap samples, averaged
Each tree sees a bootstrap sample (n rows drawn **with replacement**, about 63.2% unique rows).
The ~36.8% left out are that tree's **out-of-bag (OOB)** rows: a free validation set.

**Predict first:** will the OOB MAE be close to the valid MAE?"""),
    code("""bag = BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5), n_estimators=60, oob_score=True,
                       n_jobs=-1, random_state=SEED).fit(Xt, y_tr)
oob_mae = mae(y_tr, bag.oob_prediction_)
rows.append(score(valid, bag.predict(Xv), "bagging (60 trees)"))
print(f"OOB MAE {oob_mae:.1f} bps vs valid MAE {rows[-1]['mae_bps']:.1f} bps"); table(rows)"""),
    md("""**Concept + project twist:** in class OOB error is a good stand-in for test error. Here it is far too
optimistic: an OOB row's sister rows (same bond, other months; same issuer's other bonds) were in the bootstrap
sample, so the model has nearly seen it. OOB assumes rows are independent. Panel data breaks that, which is why
the project validates on later months only."""),
    md("""## 2. Random forest: bagging + a random subset of features at each split
`max_features=1.0` is plain bagging. Smaller `max_features` decorrelates the trees, so averaging removes more variance.

**Predict first:** which `max_features` wins?"""),
    code("""rf_rows = []
for mf in [1.0, 0.5, 0.33, "sqrt"]:
    rf = RandomForestRegressor(n_estimators=150, max_features=mf, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Xt, y_tr)
    r = score(valid, rf.predict(Xv), f"random forest max_features={mf}"); rf_rows.append(r)
best_rf_row = min(rf_rows, key=lambda r: r["mae_bps"]); rows.append(best_rf_row)
table(rf_rows)"""),
    code("""mf_best = best_rf_row["model"].split("=")[1]; mf_best = mf_best if mf_best == "sqrt" else float(mf_best)
rf = RandomForestRegressor(n_estimators=300, max_features=mf_best, min_samples_leaf=5, n_jobs=-1, random_state=SEED,
                           warm_start=True)
ntree = []
for n in [10, 25, 50, 100, 200, 300]:
    rf.set_params(n_estimators=n); rf.fit(Xt, y_tr); ntree.append({"trees": n, "valid_mae": mae(y_va, rf.predict(Xv))})
ntree = pd.DataFrame(ntree); print(ntree.round(2).to_string(index=False))"""),
    md("""**Concept:** more trees never overfits a random forest; the error just flattens. `n_estimators` is a
compute budget, not a complexity knob. The complexity knobs are depth / `min_samples_leaf` / `max_features`."""),
    md("""## 3. Feature importance: impurity vs permutation
**Predict first:** top three drivers of a bond's spread?"""),
    code("""from sklearn.inspection import permutation_importance
imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
vs = np.random.default_rng(SEED).choice(len(Xv), size=min(8000, len(Xv)), replace=False)
perm = permutation_importance(rf, Xv[vs], y_va.iloc[vs], n_repeats=3, random_state=SEED, n_jobs=-1,
                              scoring="neg_mean_absolute_error")
perm = pd.Series(perm.importances_mean, index=names).sort_values(ascending=False)
pd.DataFrame({"impurity (train)": imp.head(12).round(3), "permutation on valid (bps of MAE)": perm.head(12).round(2)})"""),
    md("""**Concept:** impurity importance is computed on training data and favours continuous, high-cardinality
features. Permutation importance shuffles one column on held-out data and measures how much the error rises:
slower, but it measures what the model actually uses out of sample."""),
    md("""## 4. Boosting: trees in sequence, each fixing the last one's errors
Gradient boosting fits each new shallow tree to the **residuals** (the negative gradient of the loss) and adds a
shrunken copy: `F_m = F_{m-1} + learning_rate x tree_m`.

**Predict first:** with learning rate 0.3 vs 0.05, which one needs more trees, and which overfits sooner?"""),
    code("""sub = np.random.default_rng(SEED).choice(len(Xt), size=min(50000, len(Xt)), replace=False)
stage = {}
for lr in [0.3, 0.1, 0.05]:
    gb = GradientBoostingRegressor(n_estimators=400, learning_rate=lr, max_depth=4, subsample=0.8,
                                   random_state=SEED).fit(Xt[sub], y_tr.iloc[sub])
    stage[lr] = [mae(y_va, p) for p in gb.staged_predict(Xv)]
fig, ax = plt.subplots(figsize=(7, 3.5))
for lr, s in stage.items(): ax.plot(range(1, len(s) + 1), s, label=f"learning rate {lr}")
ax.axhline(rows[0]["mae_bps"], ls="--", c="grey", label="matrix pricing"); ax.set_xlabel("trees"); ax.set_ylabel("valid MAE bps"); ax.legend(); plt.show()
{lr: (int(np.argmin(s)) + 1, round(min(s), 2)) for lr, s in stage.items()}"""),
    md("""**Concept:** boosting reduces **bias** (shallow trees, added up), bagging/RF reduce **variance** (deep trees,
averaged). Boosting *can* overfit with too many trees, so the number of trees is tuned together with the learning
rate (small rate + more trees usually generalises better). AdaBoost reweights misclassified rows instead of fitting
residuals."""),
    code("""hgb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                    random_state=SEED).fit(Xt, y_tr)
rows.append(score(valid, hgb.predict(Xv), "gradient boosting (hist, 600 x 0.05)"))
best_row = min(rows[1:], key=lambda r: r["mae_bps"])
gain = {g: 1 - best_row[f"mae_{g}"] / rows[0][f"mae_{g}"] for g in ["IG", "HY"]}
print("best model:", best_row["model"], "| MAE gain vs matrix pricing:", {k: f"{v:.1%}" for k, v in gain.items()})
table(rows)"""),
    code("""save("r2_ensembles", {"table": rows, "oob_mae": oob_mae, "rf_max_features": rf_rows, "n_trees": ntree.to_dict("records"),
                      "importance_impurity": imp.head(10).to_dict(), "importance_permutation": perm.head(10).to_dict(),
                      "boost_best": {str(k): [int(np.argmin(s)) + 1, float(min(s))] for k, s in stage.items()},
                      "gain_vs_matrix": gain})"""),
]

# ---------------------------------------------------------------- 3. KNN
R3 = [
    md("""# R3 - KNN: pricing a bond from its nearest comparables

**Course topic:** K-nearest neighbours: distance, scaling, choice of k, weighting.
**Project question:** a pricing desk already uses "comparable bonds". KNN is that idea with a learned notion of
"comparable": the k bonds closest in feature space, and the prediction is their average spread."""),
    code(SETUP),
    code("""from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error as mae
tr = sample(train.loc[y_tr.index], 40000); Xs, ys = xy(tr)
print(len(Xs), "training rows (a sample: KNN stores every row and compares each query with all of them)")"""),
    md("""## 1. Scaling is not optional
**Predict first:** without scaling, which feature dominates the distance? (Look at the units: `hy_oas_bps` is in
the hundreds, `coupon_pct` around 5, `vol_60d` around 0.3.)"""),
    code("""res = {}
for scale in [False, True]:
    m = Pipeline([("pre", make_preprocessor(scale=scale)), ("knn", KNeighborsRegressor(n_neighbors=10, n_jobs=-1))]).fit(Xs, ys)
    res["scaled" if scale else "raw units"] = mae(y_va, m.predict(X_va))
pd.Series(res, name="valid MAE bps").round(2)"""),
    md("""**Concept:** KNN, neural networks and anything distance- or gradient-based need standardised features.
Trees and Naive Bayes (Gaussian, per feature) do not. Fit the scaler on train only, then apply it to valid."""),
    md("""## 2. Choosing k
Small k = low bias, high variance (one odd neighbour moves the price). Large k = smooth, eventually just the average.
**Predict first:** where is the sweet spot, and does `weights="distance"` help?"""),
    code("""pre = make_preprocessor(scale=True).fit(Xs); Zs = pre.transform(Xs); Zv = pre.transform(X_va)
kc = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    for w in ["uniform", "distance"]:
        knn = KNeighborsRegressor(n_neighbors=k, weights=w, n_jobs=-1).fit(Zs, ys)
        kc.append({"k": k, "weights": w, "valid_mae": mae(y_va, knn.predict(Zv))})
kc = pd.DataFrame(kc).pivot(index="k", columns="weights", values="valid_mae"); print(kc.round(2))
kc.plot(marker="o", logx=True, figsize=(7, 3.5), ylabel="valid MAE bps"); plt.show()"""),
    code("""kbest = int(kc["distance"].idxmin())
for metric in ["euclidean", "manhattan"]:
    knn = KNeighborsRegressor(n_neighbors=kbest, weights="distance", metric=metric, n_jobs=-1).fit(Zs, ys)
    rows.append(score(valid, knn.predict(Zv), f"KNN k={kbest}, distance-weighted, {metric}"))
table(rows)"""),
    md("""## 3. Look at the comparables
KNN is easy to explain to a pricing desk: "these are the five bonds we priced it from"."""),
    code("""knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1).fit(Zs, ys)
i = int(np.argmin(np.abs(y_va - y_va.median())))
dist, idx = knn.kneighbors(Zv[i:i + 1])
show = ["issuer_ticker", "sector", "grade_bucket", "maturity_bucket", "coupon_pct", "vol_60d", "book_leverage", "spread_bps"]
print("query bond:"); display(valid.loc[[i], show])
nb = tr.iloc[idx[0]][show].assign(distance=dist[0].round(2)); display(nb)
print(f"prediction = mean of the 5 spreads = {nb.spread_bps.mean():.1f} bps; actual = {y_va[i]:.1f} bps")"""),
    md("""**Concept:** KNN is a *lazy learner*: "training" only stores the data, all the work happens at prediction time
(cost grows with the number of training rows). With many features, distances concentrate and all points look
equally far apart: the **curse of dimensionality**. Collaborative filtering in class used the same neighbour idea on
users x items."""),
    code("""save("r3_knn", {"table": rows, "scaling": res, "k_curve": kc.reset_index().to_dict("records"), "k_best": kbest})"""),
]

# ---------------------------------------------------------------- 4. Naive Bayes
R4 = [
    md("""# R4 - Naive Bayes: will this bond's spread widen next month?

**Course topic:** Bayes' rule, the naive independence assumption, Laplace smoothing, priors.
**Project question:** Naive Bayes only classifies, so we use the annex target `widens_25bp` (spread up 25 bp or more by
next month-end). About 4% of rows say yes."""),
    code(SETUP.replace('base = baseline_valid(valid)\nrows = [score(valid, base, "matrix pricing (same date)")]\n', "")
         .replace("\ntable(rows)", "")),
    code("""Xc_tr, yc_tr = xy(train, "widens_25bp"); Xc_va, yc_va = xy(valid, "widens_25bp")
yc_tr = yc_tr.astype(int); yc_va = yc_va.astype(int)
trc = train.loc[Xc_tr.index]; vac = valid.loc[Xc_va.index]
print(f"P(widen) train = {yc_tr.mean():.4f}, valid = {yc_va.mean():.4f}")"""),
    md("""## 1. Bayes by hand on two features
Discretise two features into words a desk would use: `grade_bucket` (IG / HY) and VIX regime (calm < 15 <= normal < 20
<= stressed). Then

P(widen | HY, stressed) = P(HY | widen) P(stressed | widen) P(widen) / P(HY, stressed)

with the naive assumption that grade and VIX regime are independent **given** the class.

**Predict first:** is a HY bond on a stressed day more or less than 4% likely to widen?"""),
    code("""def regime(v): return pd.cut(v, [-np.inf, 15, 20, np.inf], labels=["calm", "normal", "stressed"])
d = pd.DataFrame({"grade": trc["grade_bucket"].to_numpy(), "vix": regime(trc["vix"]).to_numpy(), "y": yc_tr.to_numpy()})
prior = d.y.value_counts(normalize=True).sort_index()
alpha = 1  # Laplace smoothing
def likelihood(col, val, cls):
    sub = d[d.y == cls]; k = d[col].nunique()
    return (np.sum(sub[col] == val) + alpha) / (len(sub) + alpha * k)
score_ = {c: prior[c] * likelihood("grade", "HY", c) * likelihood("vix", "stressed", c) for c in [0, 1]}
post1 = score_[1] / (score_[0] + score_[1])
print("priors:", prior.round(4).to_dict())
for c in [0, 1]:
    print(f"class {c}: P(HY|c)={likelihood('grade','HY',c):.4f}  P(stressed|c)={likelihood('vix','stressed',c):.4f}  numerator={score_[c]:.6f}")
print(f"P(widen | HY, stressed) = {post1:.4f}")"""),
    code("""from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder().fit(d[["grade", "vix"]].astype(str))
cnb = CategoricalNB(alpha=1).fit(enc.transform(d[["grade", "vix"]].astype(str)), d.y)
q = enc.transform(pd.DataFrame({"grade": ["HY"], "vix": ["stressed"]}))
print("sklearn CategoricalNB:", cnb.predict_proba(q)[0, 1].round(4), "| by hand:", round(post1, 4))
emp = d[(d.grade == "HY") & (d.vix == "stressed")].y.mean()
print(f"actual share of widenings in that cell (train): {emp:.4f}")"""),
    md("""**Concept:** the denominator P(HY, stressed) is the same for both classes, so you can compare numerators and
normalise at the end. Laplace smoothing adds alpha to every count so an unseen value never gives probability 0
(which would zero the whole product). If the hand result and the actual cell share differ, the "naive" independence
assumption is the reason."""),
    md("""## 2. Gaussian Naive Bayes on all numeric features
Each feature is modelled as a normal distribution per class. **Predict first:** with a 4% prior, how often will it
predict "widen"?"""),
    code("""from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
pre = make_preprocessor(scale=False, one_hot=True)
Ct = pre.fit_transform(Xc_tr); Cv = pre.transform(Xc_va)
def clf_scores(name, p, proba):
    return {"model": name, "accuracy": accuracy_score(yc_va, p), "recall": recall_score(yc_va, p, zero_division=0),
            "precision": precision_score(yc_va, p, zero_division=0), "roc_auc": roc_auc_score(yc_va, proba),
            "pr_auc": average_precision_score(yc_va, proba), "predicted_1_share": float(np.mean(p))}
crow = [clf_scores("always 0", np.zeros(len(yc_va), int), np.zeros(len(yc_va)) + yc_tr.mean())]
gnb = GaussianNB().fit(Ct, yc_tr); crow.append(clf_scores("Gaussian NB", gnb.predict(Cv), gnb.predict_proba(Cv)[:, 1]))
gnb_eq = GaussianNB(priors=[0.5, 0.5]).fit(Ct, yc_tr)
crow.append(clf_scores("Gaussian NB, equal priors", gnb_eq.predict(Cv), gnb_eq.predict_proba(Cv)[:, 1]))
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1,
                            random_state=SEED).fit(Ct, yc_tr)
crow.append(clf_scores("random forest (balanced)", rf.predict(Cv), rf.predict_proba(Cv)[:, 1]))
pd.DataFrame(crow).round(3)"""),
    md("""**Concept:**
- With a tiny prior, NB rarely predicts the rare class; changing `priors` moves the decision, not the likelihoods.
- NB's probabilities are often badly calibrated (pushed towards 0 or 1) because correlated features are counted as
  independent evidence: here the three volatility windows and the two credit indices say the same thing several times.
- Compare models on ROC AUC / PR AUC (threshold-free). The PR AUC of a random guess equals the positive share (~0.04).
- NB is fast, needs little data and works well on text (word counts are many, weak, roughly independent signals:
  the SMS spam and tweets notebooks)."""),
    code("""save("r4_naive_bayes", {"hand_posterior_HY_stressed": post1, "sklearn_posterior": float(cnb.predict_proba(q)[0, 1]),
                        "cell_share": float(emp), "prior_widen": float(prior[1]), "classifiers": crow})"""),
]

# ---------------------------------------------------------------- 5. Neural network
R5 = [
    md("""# R5 - A neural network (MLP) on bond spreads

**Course topic:** Deep learning sessions: neurons, activations, layers, loss, gradient descent, epochs, overfitting.
**Project question:** does a small multilayer perceptron beat the trees on a tabular problem?"""),
    code(SETUP),
    code("""from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error as mae
tr = sample(train.loc[y_tr.index], 60000); Xs, ys = xy(tr)
pre = make_preprocessor(scale=True).fit(Xs); Zs = pre.transform(Xs); Zv = pre.transform(X_va)
print(Zs.shape, "- inputs after one-hot and missing flags")"""),
    md("""## 1. Count the parameters before you train
A dense layer with n_in inputs and n_out neurons has n_in x n_out weights + n_out biases.
**Predict first:** how many parameters does a 64-32-1 network have on these inputs?"""),
    code("""n_in = Zs.shape[1]; layers = [n_in, 64, 32, 1]
params = sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))
print(f"{' -> '.join(map(str, layers))}: {params:,} parameters for {len(Zs):,} training rows")"""),
    md("""## 2. Train, watch the loss, stop early
`early_stopping=True` holds out 10% of the training rows and stops when their score stops improving
(`n_iter_no_change` epochs). `alpha` is an L2 penalty on the weights. Scale the target too: the network learns faster
on numbers near 1 than on 300 bps."""),
    code("""from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
mlps = {}
for name, kw in {"64-32 relu": dict(hidden_layer_sizes=(64, 32), activation="relu"),
                 "64-32 tanh": dict(hidden_layer_sizes=(64, 32), activation="tanh"),
                 "128-64-32 relu, alpha 1e-3": dict(hidden_layer_sizes=(128, 64, 32), activation="relu", alpha=1e-3)}.items():
    m = TransformedTargetRegressor(MLPRegressor(solver="adam", learning_rate_init=1e-3, batch_size=256, max_iter=200,
                                                early_stopping=True, n_iter_no_change=10, random_state=SEED, **kw),
                                   transformer=StandardScaler()).fit(Zs, ys)
    mlps[name] = m; rows.append(score(valid, m.predict(Zv), f"MLP {name}"))
    print(name, "- epochs run:", m.regressor_.n_iter_)
table(rows)"""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for name, m in mlps.items():
    ax[0].plot(m.regressor_.loss_curve_, label=name); ax[1].plot(m.regressor_.validation_scores_, label=name)
ax[0].set_title("training loss (scaled target)"); ax[1].set_title("early-stopping R^2 (held-out 10%)")
for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=8)
plt.show()"""),
    md("""**Concept:**
- One epoch = one pass over all training rows; with `batch_size=256` that is n/256 gradient steps.
- ReLU = max(0, z) avoids the vanishing gradients of sigmoid/tanh in deep nets; the output neuron of a regression net
  is linear (no activation).
- Training loss always falls; the held-out curve shows when the network starts to overfit, which is what early
  stopping and L2 (`alpha`) / dropout fight.
- On tabular data like this, boosted trees usually match or beat an MLP. CNNs need data with spatial structure
  (neighbouring pixels); a row of bond features has none, so a convolution has nothing to slide over."""),
    code("""save("r5_neural_net", {"table": rows, "params_64_32": params, "n_inputs": n_in})"""),
]

# ---------------------------------------------------------------- 6. Summary
R6 = [
    md("""# R6 - Summary: every model on the same valid months

Reads `results/*.json` written by R1-R5. Fill in the "why" column yourself: that is the takeaway."""),
    code("""import json, pandas as pd
from pathlib import Path
res = {p.stem: json.loads(p.read_text()) for p in sorted(Path("results").glob("r*.json"))}
rows = {}
for k, v in res.items():
    for r in v.get("table", []): rows[r["model"]] = r
t = pd.DataFrame(rows.values()).set_index("model")[["mae_bps", "mae_IG", "mae_HY", "rmse_bps", "r2"]].sort_values("mae_bps")
base = t.loc["matrix pricing (same date)"]
t["gain_IG"] = 1 - t.mae_IG / base.mae_IG; t["gain_HY"] = 1 - t.mae_HY / base.mae_HY
t.round(3)"""),
    md("""| Model | Bias / variance story | Scaling? | Why it did (or did not) win here |
|---|---|---|---|
| Pruned tree | | No | |
| Bagging / RF | | No | |
| Gradient boosting | | No | |
| KNN | | Yes | |
| MLP | | Yes | |
| Naive Bayes (classifier) | | No | |"""),
]


def main():
    books = {"R1_trees.ipynb": R1, "R2_ensembles.ipynb": R2, "R3_knn.ipynb": R3, "R4_naive_bayes.ipynb": R4,
             "R5_neural_net.ipynb": R5, "R6_summary.ipynb": R6}
    # Pick notebooks by prefix (e.g. R1 R2); none = all. An existing notebook is only rewritten when it is picked
    # by name or --force is given, so a run you have read is never wiped by a rebuild of another notebook.
    picked = [a.upper() for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    todo = [n for n in books if not picked or n.split("_")[0] in picked]
    for name in todo:
        path = HERE / name
        if path.exists() and not picked and not force:
            print("kept", name, "(exists; name it or pass --force to rewrite)")
            continue
        nbf.write(notebook(books[name]), path)
        print("wrote", name)
    if "--execute" in sys.argv:
        for name in todo:
            print("running", name, flush=True)
            subprocess.run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=3600", str(HERE / name)], check=True, cwd=HERE)


if __name__ == "__main__":
    main()
