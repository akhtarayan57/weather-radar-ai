import numpy as np


class Subset:
    """Wraps a model so it only sees chosen feature columns."""

    def __init__(self, base, feats):
        self.base, self.feats = base, list(feats)

    def fit(self, X, y):
        self.base.fit(X[self.feats], y)
        return self

    def predict(self, X):
        return self.base.predict(X[self.feats])


class BlendModel:
    """w * model + (1 - w) * persistence, in log space. w is chosen on validation data."""

    def __init__(self, base, w):
        self.base, self.w = base, float(w)

    def predict(self, X):
        return self.w * self.base.predict(X) + (1 - self.w) * np.asarray(X["pm_lag0"])
