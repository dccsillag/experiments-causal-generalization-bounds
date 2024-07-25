import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, clone  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore


class DummyModel:
    def predict(self, X):
        return np.zeros(X.shape[0])


class TLearner:
    def __init__(
        self,
        *,
        base_learner: BaseEstimator,
    ):
        self.base_learner = base_learner

    def fit(self, X, T, Y, propensities=None):
        X0 = X[T == 0, ...]
        Y0 = Y[T == 0, ...]
        X1 = X[T == 1, ...]
        Y1 = Y[T == 1, ...]

        if propensities is not None:
            W1 = (1 / propensities[T == 1],)
            W0 = (1 / (1 - propensities[T == 0]),)
        else:
            W1 = ()
            W0 = ()

        if np.sum(T == 0) > 0:
            self.model0 = clone(self.base_learner).fit(X0, Y0, *W0)
        else:
            self.model0 = DummyModel()
        if np.sum(T == 1) > 0:
            self.model1 = clone(self.base_learner).fit(X1, Y1, *W1)
        else:
            self.model1 = DummyModel()

        return self

    def predict(self, X):
        return self.model1.predict(X) - self.model0.predict(X)


class SLearner:
    def __init__(
        self,
        *,
        base_learner: BaseEstimator,
    ):
        self.base_learner = base_learner

    def fit(self, X, T, Y, propensities=None):
        XT = np.concatenate((X, T[:, None]), axis=1)

        if propensities is not None:
            W = (1 / np.where(T == 1, propensities, 1 - propensities),)
        else:
            W = ()

        self.model = clone(self.base_learner).fit(XT, Y, *W)

        return self

    def individual_predict(self, X, *, t):
        return self.model.predict(
            np.concatenate((X, t * np.ones((X.shape[0], 1))), axis=1)
        )

    def predict(self, X):
        return self.individual_predict(X, t=1) - self.individual_predict(X, t=0)


class XLearner:
    def __init__(
        self,
        *,
        base_learner: BaseEstimator,
    ):
        self.base_learner = base_learner

    def fit(self, X, T, Y, propensities=None):
        X0 = X[T == 0, ...]
        Y0 = Y[T == 0, ...]
        X1 = X[T == 1, ...]
        Y1 = Y[T == 1, ...]

        if propensities is not None:
            W1 = (1 / propensities[T == 1],)
            W0 = (1 / (1 - propensities[T == 0]),)
        else:
            W1 = ()
            W0 = ()

        self.propensity_model = CalibratedClassifierCV(
            RandomForestClassifier(random_state=0)
        ).fit(X, T)

        if np.sum(T == 0) > 0:
            self.model0 = clone(self.base_learner).fit(X0, Y0, *W0)
        else:
            self.model0 = DummyModel()
        if np.sum(T == 1) > 0:
            self.model1 = clone(self.base_learner).fit(X1, Y1, *W1)
        else:
            self.model1 = DummyModel()

        pseudo_Y0 = self.model1.predict(X0) - Y0
        pseudo_Y1 = Y1 - self.model0.predict(X1)

        if np.sum(T == 0) > 0:
            self.model01 = clone(self.base_learner).fit(X0, pseudo_Y0, *W0)
        else:
            self.model01 = DummyModel()
        if np.sum(T == 1) > 0:
            self.model10 = clone(self.base_learner).fit(X1, pseudo_Y1, *W1)
        else:
            self.model10 = DummyModel()

        return self

    def predict(self, X):
        e = self.propensity_model.predict_proba(X)[:, 1]
        return self.model10.predict(X) * e + self.model01.predict(X) * (1 - e)
