"""Runs every experiment and saves plots + results.json to results/."""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.decision_tree import DecisionTree
from src.utils import metrics

SEED = 42
data = load_breast_cancer()
X, y, names = data.data, data.target, list(data.feature_names)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
res = {"dataset": "Breast Cancer Wisconsin", "n_samples": len(y), "n_features": X.shape[1],
       "class_balance": {data.target_names[i]: int((y == i).sum()) for i in range(2)}}

# 1. Head-to-head, depth 5
t0 = time.time(); mine = DecisionTree("gini", max_depth=5, min_samples_split=2).fit(Xtr, ytr); t_mine = time.time() - t0
t0 = time.time(); sk = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=SEED).fit(Xtr, ytr); t_sk = time.time() - t0
res["from_scratch"] = metrics(yte, mine.predict(Xte)); res["sklearn"] = metrics(yte, sk.predict(Xte))
res["fit_seconds"] = {"from_scratch": t_mine, "sklearn": t_sk}
res["prediction_agreement"] = float((mine.predict(Xte) == sk.predict(Xte)).mean())

# criterion comparison
for c in ("gini", "entropy"):
    m = DecisionTree(c, max_depth=5).fit(Xtr, ytr)
    res[f"from_scratch_{c}_acc"] = metrics(yte, m.predict(Xte))["accuracy"]

# 2. Depth sweep
depths = list(range(1, 13)); tr_a, te_a, sk_te = [], [], []
for d in depths:
    m = DecisionTree("gini", max_depth=d).fit(Xtr, ytr)
    tr_a.append((m.predict(Xtr) == ytr).mean()); te_a.append((m.predict(Xte) == yte).mean())
    sk_te.append(DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr).score(Xte, yte))
res["depth_sweep"] = {"depths": depths, "train": tr_a, "test": te_a, "sklearn_test": sk_te}
res["best_depth_test"] = depths[int(np.argmax(te_a))]
plt.figure(figsize=(7, 4.5))
plt.plot(depths, tr_a, "o-", label="Train (scratch)"); plt.plot(depths, te_a, "s-", label="Test (scratch)")
plt.plot(depths, sk_te, "^--", label="Test (sklearn)", alpha=.7)
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.title("Depth vs. accuracy: overfitting check")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig("results/depth_vs_accuracy.png", dpi=150); plt.close()

# min_samples_split sweep
mss = [2, 5, 10, 20, 40]
res["min_samples_split_sweep"] = {str(s): float((DecisionTree("gini", min_samples_split=s).fit(Xtr, ytr).predict(Xte) == yte).mean()) for s in mss}

# 3. Confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, (title, pred) in zip(ax, [("From scratch", mine.predict(Xte)), ("scikit-learn", sk.predict(Xte))]):
    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=data.target_names).plot(ax=a, colorbar=False, cmap="Blues")
    a.set_title(title)
plt.tight_layout(); plt.savefig("results/confusion_matrices.png", dpi=150); plt.close()

# 4. Feature importance (top 10, both)
idx = np.argsort(mine.feature_importances_)[::-1][:10]
res["top_features_scratch"] = {names[i]: float(mine.feature_importances_[i]) for i in idx}
plt.figure(figsize=(8, 5))
w = 0.4; pos = np.arange(10)
plt.barh(pos - w/2, mine.feature_importances_[idx][::-1], w, label="From scratch")
plt.barh(pos + w/2, sk.feature_importances_[idx][::-1], w, label="scikit-learn")
plt.yticks(pos, [names[i] for i in idx][::-1]); plt.xlabel("Importance"); plt.title("Top 10 feature importances")
plt.legend(); plt.tight_layout(); plt.savefig("results/feature_importance.png", dpi=150); plt.close()

# 5. Tree diagram (sklearn depth 3 for readability) + text dump of scratch tree
plt.figure(figsize=(16, 7))
plot_tree(DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, ytr), feature_names=names,
          class_names=list(data.target_names), filled=True, rounded=True, fontsize=8)
plt.tight_layout(); plt.savefig("results/tree_diagram.png", dpi=150); plt.close()
open("results/scratch_tree.txt", "w").write(mine.export_text(names, list(data.target_names)))

# 6. Random Forest extension
rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(Xtr, ytr)
res["random_forest"] = metrics(yte, rf.predict(Xte))

json.dump(res, open("results/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
