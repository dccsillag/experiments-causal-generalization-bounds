from dataclasses import dataclass

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from ForestDiffusion import ForestDiffusionModel  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from causallib.datasets.data_loader import load_acic16  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


@dataclass
class Dataset:
    X: np.ndarray
    T: np.ndarray
    propensities: np.ndarray
    Y: np.ndarray
    Y1: np.ndarray
    Y0: np.ndarray

    def train_test_split(
        self, *, test_size: float = 0.25
    ) -> tuple["Dataset", "Dataset"]:
        (
            X_train,
            X_test,
            T_train,
            T_test,
            propensities_train,
            propensities_test,
            Y_train,
            Y_test,
            Y1_train,
            Y1_test,
            Y0_train,
            Y0_test,
        ) = train_test_split(
            self.X,
            self.T,
            self.propensities,
            self.Y,
            self.Y1,
            self.Y0,
            test_size=test_size,
            random_state=0,
        )

        return (
            Dataset(
                X=X_train,
                T=T_train,
                propensities=propensities_train,
                Y=Y_train,
                Y1=Y1_train,
                Y0=Y0_train,
            ),
            Dataset(
                X=X_test,
                T=T_test,
                propensities=propensities_test,
                Y=Y_test,
                Y1=Y1_test,
                Y0=Y0_test,
            ),
        )


def load_nearrct() -> Dataset:
    rng = np.random.default_rng(0)

    df_raw = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        header=None,
    )
    col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]
    for i in range(1, 26):
        col.append(f"x{i}")
    df_raw.columns = col

    T_raw = df_raw["treatment"].to_numpy()
    Y_raw = df_raw["y_factual"].to_numpy()
    X_raw = df_raw.drop(
        ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"], axis=1
    ).to_numpy()

    X0 = X_raw[T_raw == 0, ...]
    Y0 = Y_raw[T_raw == 0, ...]
    X1 = X_raw[T_raw == 1, ...]
    Y1 = Y_raw[T_raw == 1, ...]

    gen_x = ForestDiffusionModel(
        X_raw,
        n_t=10,
        duplicate_K=1,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=[],
        diffusion_type="flow",
        n_jobs=-4,
    )

    treatment = CalibratedClassifierCV(
        RandomForestClassifier(random_state=0, n_jobs=-2), cv=3
    ).fit(X_raw, T_raw)
    model0 = GaussianProcessRegressor(random_state=0).fit(X0, Y0)
    model1 = GaussianProcessRegressor(random_state=0).fit(X1, Y1)

    n_X_features = X_raw.shape[1]
    n = 4_000

    X_out = gen_x.generate(batch_size=n)
    Y0_out = model0.sample_y(X_out, n_samples=1, random_state=rng.integers(1e6))[:, 0]
    Y1_out = model1.sample_y(X_out, n_samples=1, random_state=rng.integers(1e6))[:, 0]

    propensities_out = treatment.predict_proba(X_out)[:, 1]
    T_out = np.where(rng.uniform(size=X_out.shape[0]) <= propensities_out, 1, 0)

    Y_out = np.where(T_out == 1, Y1_out, Y0_out)

    assert X_out.shape == (n, n_X_features)
    assert T_out.shape == (n,)
    assert propensities_out.shape == (n,)
    assert Y1_out.shape == (n,)
    assert Y0_out.shape == (n,)
    assert Y_out.shape == (n,)

    return Dataset(
        X=X_out, T=T_out, propensities=propensities_out, Y=Y_out, Y1=Y1_out, Y0=Y0_out
    )


def load_observational() -> Dataset:
    data = load_acic16(instance=2)
    T = data["a"].to_numpy()
    Y = data["y"].to_numpy()
    X = data["X"].fillna(0).astype(float).to_numpy()
    Y1 = data["po"]["1"].fillna(0).astype(float).to_numpy()
    Y0 = data["po"]["0"].fillna(0).astype(float).to_numpy()

    treatment = CalibratedClassifierCV(
        RandomForestClassifier(random_state=0, n_jobs=-2), cv=3
    ).fit(X, T)
    propensities = treatment.predict_proba(X)[:, 1]

    return Dataset(X=X, T=T, Y=Y, propensities=propensities, Y1=Y1, Y0=Y0)


def load_hiddenconfounding() -> Dataset:
    data = load_acic16(instance=2)
    T = data["a"].to_numpy()
    # Y = data["y"].to_numpy()
    X = data["X"].fillna(0).astype(float).to_numpy()
    Y1 = data["po"]["1"].fillna(0).astype(float).to_numpy()
    Y0 = data["po"]["0"].fillna(0).astype(float).to_numpy()

    treatment = CalibratedClassifierCV(
        RandomForestClassifier(random_state=0, n_jobs=-2), cv=3
    ).fit(X, T)
    propensities = treatment.predict_proba(X)[:, 1]

    # Here, we consider the raw $T$ to be a hidden confounder, by making the output $T$ into something else and tweaking the outcome based on the raw $T$.
    T_out = T

    Y1 = np.where(T == 1, Y1, Y1 - 20)
    Y = np.where(T_out == 1, Y1, Y0)
    rectified_propensties = T_out

    return Dataset(X=X, T=T_out, Y=Y, propensities=rectified_propensties, Y1=Y1, Y0=Y0)
