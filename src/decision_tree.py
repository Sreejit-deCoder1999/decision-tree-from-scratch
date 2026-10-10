"""Decision tree classifier implemented from scratch with NumPy."""
import numpy as np


def gini(y):
    """Gini impurity: 1 - sum(p_k^2)."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p ** 2)


def entropy(y):
    """Shannon entropy: -sum(p_k * log2 p_k)."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n", "impurity")

    def __init__(self, value, n, impurity, feature=None, threshold=None, left=None, right=None):
        self.value, self.n, self.impurity = value, n, impurity
        self.feature, self.threshold, self.left, self.right = feature, threshold, left, right

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._impurity = gini if criterion == "gini" else entropy

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root_ = self._build(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _best_split(self, X, y):
        """Search every feature and every midpoint threshold for the max information gain."""
        n, parent = len(y), self._impurity(y)
        best = (None, None, 0.0)  # feature, threshold, gain
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j])
            xs, ys = X[order, j], y[order]
            # incremental class counts keep each feature O(n) after sorting
            left_counts = np.zeros(self.n_classes_)
            right_counts = np.bincount(ys, minlength=self.n_classes_).astype(float)
            for i in range(n - 1):
                c = ys[i]
                left_counts[c] += 1
                right_counts[c] -= 1
                if xs[i] == xs[i + 1]:
                    continue
                nl, nr = i + 1, n - i - 1
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                pl, pr = left_counts / nl, right_counts / nr
                if self.criterion == "gini":
                    il, ir = 1 - np.sum(pl ** 2), 1 - np.sum(pr ** 2)
                else:
                    il = -np.sum(pl[pl > 0] * np.log2(pl[pl > 0]))
                    ir = -np.sum(pr[pr > 0] * np.log2(pr[pr > 0]))
                gain = parent - (nl / n) * il - (nr / n) * ir
                if gain > best[2]:
                    best = (j, (xs[i] + xs[i + 1]) / 2, gain)
        return best

    def _build(self, X, y, depth):
        n = len(y)
        imp = self._impurity(y)
        value = np.bincount(y, minlength=self.n_classes_)
        stop = (imp == 0 or n < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth))
        if not stop:
            j, thr, gain = self._best_split(X, y)
            if j is not None:
                self.feature_importances_[j] += gain * n
                mask = X[:, j] <= thr
                left = self._build(X[mask], y[mask], depth + 1)
                right = self._build(X[~mask], y[~mask], depth + 1)
                return Node(value, n, imp, j, thr, left, right)
        return Node(value, n, imp)

    def _predict_one(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return int(np.argmax(node.value))

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X, dtype=float)])

    def get_depth(self):
        def d(node):
            return 0 if node.is_leaf else 1 + max(d(node.left), d(node.right))
        return d(self.root_)

    def export_text(self, feature_names=None, class_names=None):
        lines = []
        def rec(node, indent):
            pad = "|   " * indent
            if node.is_leaf:
                c = int(np.argmax(node.value))
                lines.append(f"{pad}class: {class_names[c] if class_names else c}  (n={node.n})")
                return
            name = feature_names[node.feature] if feature_names else f"x[{node.feature}]"
            lines.append(f"{pad}{name} <= {node.threshold:.3f}")
            rec(node.left, indent + 1)
            lines.append(f"{pad}{name} >  {node.threshold:.3f}")
            rec(node.right, indent + 1)
        rec(self.root_, 0)
        return "\n".join(lines)
