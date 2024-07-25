from typing import Callable
from dataclasses import dataclass
from argparse import ArgumentParser

from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from geomloss import SamplesLoss  # type: ignore
import torch  # type: ignore

from data import load_nearrct, load_observational, load_hiddenconfounding, Dataset
from metalearners import TLearner


parser = ArgumentParser()
parser.add_argument("--without-prior-work", action="store_true")
parser.add_argument("--more-losses", action="store_true")
parser.add_argument(
    "--what", choices=["outcome", "tlearner", "slearner", "xlearner"], default="outcome"
)
parser.add_argument("-o", "--output", default="out.png")
args = parser.parse_args()


rng = np.random.default_rng(0)


def mean_squared_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_absolute_loss(y_true, y_pred):
    return np.abs(y_true - y_pred)


def quantile_loss(y_true, y_pred):
    ALPHA = 0.8
    diff = y_true - y_pred
    return diff * ALPHA * (diff >= 0) - diff * (1 - ALPHA) * (diff < 0)


def zero_one_loss(y_true, y_pred):
    # `Y_train_all` comes from the caller's scope, disgustingly.
    mid = np.median(Y_train_all)
    y_true = y_true >= mid
    y_pred = y_pred >= mid
    return np.where(y_true == y_pred, 0, 1)


def our_bound(
    X, Y1_true, Y1_pred, T, propensity_scores, *, reweight: bool, theoric: bool, loss
):
    prob_t1 = np.mean(T)

    if reweight:
        weights = prob_t1 / (propensity_scores + 1e-11)
    else:
        weights = np.ones(X.shape[0])

    observable_risk = np.mean(weights[T == 1] * loss(Y1_true[T == 1], Y1_pred[T == 1]))
    var_1 = np.var(loss(Y1_true, Y1_pred))

    if theoric:
        chi2_1 = np.mean(((weights * propensity_scores / prob_t1) - 1) ** 2)
    else:
        chi2_loss_1 = np.mean(
            (weights * propensity_scores / prob_t1 - weights * T / prob_t1) ** 2
        )
        chi2_dev_1 = np.mean((weights * T / prob_t1 - 1) ** 2)
        chi2_1 = 2 * (chi2_loss_1 + chi2_dev_1)

    margin = lambda k: k * chi2_1 + 0.25 * var_1 / k
    best_k = np.sqrt(0.25 * var_1 / chi2_1)

    return observable_risk + margin(best_k)


EPS = 1e-4


def prior_work_bound(
    X, Y1_true, Y1_pred, Y1_pred_eps, T, propensity_scores, *, reweight: bool, loss
):
    prob_t1 = np.mean(T)

    if reweight:
        weights = prob_t1 / (propensity_scores + 1e-11)
    else:
        weights = np.ones(X.shape[0])

    observable_risk = np.mean(weights[T == 1] * loss(Y1_true[T == 1], Y1_pred[T == 1]))

    lipschitz_const = max(
        [
            np.max(
                np.abs(loss(Y1_true, Y1_pred_eps) - loss(Y1_true, Y1_pred))
                / np.sqrt(X.shape[1] * EPS**2)
            )
            for EPS in [1e-2, 1e-3, 1e-4, 1e-5]
        ]
    )  # A gross lower bound on this quantity, being generous with this benchmark here
    ipm = SamplesLoss(
        loss="sinkhorn",
        blur=0.001,  # sinkhorn with blur=0 is the Wasserstein distance. But we need something a little larger than 0 so that we don't crash... https://github.com/jeanfeydy/geomloss/pull/77
    )(torch.from_numpy(X[T == 1]), torch.from_numpy(X[T == 0]))

    return observable_risk + (1 - prob_t1) * lipschitz_const * ipm


@dataclass
class ExperimentResult:
    n: int
    proba_t1: float

    observable_risk_y1: float
    causal_risk_y1: float
    our_bound_y1: float
    our_empirical_bound_y1: float
    prior_work_bound_y1: float

    observable_risk_y0: float
    causal_risk_y0: float
    our_bound_y0: float
    our_empirical_bound_y0: float
    prior_work_bound_y0: float

    observable_risk_tlearner: float
    causal_risk_tlearner: float
    our_bound_tlearner: float
    our_empirical_bound_tlearner: float
    prior_work_bound_tlearner: float


def run_experiment(
    *, n: int, loss: Callable[[np.ndarray, np.ndarray], np.ndarray], C: float
) -> ExperimentResult:
    # NOTE: {X,T,propensities,Y,Y1,Y0}_{train,test}_all are propagated into this function's
    # scope through Python's loose/wacky/messed up scope management. They are defined right
    # before this function is used, and so remain in scope when this function is called.
    X_train = X_train_all[:n, ...]
    T_train = T_train_all[:n, ...]
    propensities_train = propensities_train_all[:n, ...]
    Y_train = Y_train_all[:n, ...]
    Y1_train = Y1_train_all[:n, ...]
    Y0_train = Y0_train_all[:n, ...]
    X_test = X_test_all[:, ...]
    T_test = T_test_all[:, ...]
    propensities_test = propensities_test_all[:, ...]
    Y_test = Y_test_all[:, ...]
    Y1_test = Y1_test_all[:, ...]
    Y0_test = Y0_test_all[:, ...]

    tlearner = TLearner(base_learner=RandomForestRegressor(random_state=0))
    tlearner.fit(X_train, T_train, Y_train)

    tau_test = tlearner.predict(X_test)
    y1_test = tlearner.model1.predict(X_test)
    y0_test = tlearner.model0.predict(X_test)
    y1_test_eps = tlearner.model1.predict(X_test + np.ones_like(X_test) * EPS)
    y0_test_eps = tlearner.model0.predict(X_test + np.ones_like(X_test) * EPS)

    # ---

    observable_risk_1 = np.mean(loss(Y1_test[T_test == 1], y1_test[T_test == 1]))
    real_risk_1 = np.mean(loss(Y1_test, y1_test))
    observable_risk_0 = np.mean(loss(Y0_test[T_test == 0], y0_test[T_test == 0]))
    real_risk_0 = np.mean(loss(Y0_test, y0_test))

    real_risk_tlearner = np.mean(loss(Y1_test - Y0_test, y1_test - y0_test))

    our_bound_y1 = our_bound(
        X_test,
        Y1_test,
        y1_test,
        T_test,
        propensities_test,
        reweight=False,
        theoric=True,
        loss=loss,
    )
    our_empirical_bound_y1 = our_bound(
        X_test,
        Y1_test,
        y1_test,
        T_test,
        propensities_test,
        reweight=False,
        theoric=False,
        loss=loss,
    )
    prior_work_bound_y1 = prior_work_bound(
        X_test,
        Y1_test,
        y1_test,
        y1_test_eps,
        T_test,
        propensities_test,
        reweight=False,
        loss=loss,
    )
    our_bound_y0 = our_bound(
        X_test,
        Y0_test,
        y0_test,
        1 - T_test,
        1 - propensities_test,
        reweight=False,
        theoric=True,
        loss=loss,
    )
    our_empirical_bound_y0 = our_bound(
        X_test,
        Y0_test,
        y0_test,
        1 - T_test,
        1 - propensities_test,
        reweight=False,
        theoric=False,
        loss=loss,
    )
    prior_work_bound_y0 = prior_work_bound(
        X_test,
        Y0_test,
        y0_test,
        y0_test_eps,
        1 - T_test,
        1 - propensities_test,
        reweight=False,
        loss=loss,
    )

    our_bound_tlearner = C * (our_bound_y1 + our_bound_y0)
    our_empirical_bound_tlearner = C * (our_empirical_bound_y1 + our_empirical_bound_y0)
    prior_work_bound_tlearner = C * (prior_work_bound_y1 + prior_work_bound_y0)

    return ExperimentResult(
        n=n,
        proba_t1=np.mean(np.concatenate((T_train, T_test))),
        observable_risk_y1=observable_risk_1,
        causal_risk_y1=real_risk_1,
        our_bound_y1=our_bound_y1,
        our_empirical_bound_y1=our_empirical_bound_y1,
        prior_work_bound_y1=prior_work_bound_y1,
        observable_risk_y0=observable_risk_0,
        causal_risk_y0=real_risk_0,
        our_bound_y0=our_bound_y0,
        our_empirical_bound_y0=our_empirical_bound_y0,
        prior_work_bound_y0=prior_work_bound_y0,
        observable_risk_tlearner=C * (observable_risk_1 + observable_risk_0),
        causal_risk_tlearner=real_risk_tlearner,
        our_bound_tlearner=our_bound_tlearner,
        our_empirical_bound_tlearner=our_empirical_bound_tlearner,
        prior_work_bound_tlearner=prior_work_bound_tlearner,
    )


DATASETS = {
    "Dataset 1: Near-RCT": load_nearrct,
    "Dataset 2: Observational": load_observational,
    "Dataset 3: Hidden Confounding": load_hiddenconfounding,
}
LOSSES = {
    "Root mean squared loss": (mean_squared_loss, np.sqrt, 2.0),
    "Mean absolute loss": (mean_absolute_loss, lambda x: x, 1.0),
}
if args.more_losses:
    LOSSES |= {
        "Quantile loss": (quantile_loss, lambda x: x, 0.8 / 0.2),
        "0-1 loss": (zero_one_loss, lambda x: x, 1.0),
    }


if args.more_losses:
    fig, axs = plt.subplots(len(LOSSES), 3, sharex="col", figsize=(12, 10))
else:
    fig, axs = plt.subplots(len(LOSSES), 3, sharex="col", figsize=(12, 4.5))

for j, (pretty_dataset_name, dgp) in enumerate(DATASETS.items()):
    axs[-1, j].set_xlabel("Number of training samples")
    axs[0, j].set_title(pretty_dataset_name)
for i, (pretty_loss_name, (loss, loss_bijector, C)) in enumerate(LOSSES.items()):
    axs[i, 0].set_ylabel(pretty_loss_name)

for j, (pretty_dataset_name, dgp) in enumerate(DATASETS.items()):
    print(f"=> {pretty_dataset_name}")
    full_dataset: Dataset = dgp()  # type: ignore
    train_dataset, test_dataset = full_dataset.train_test_split()
    X_train_all = train_dataset.X
    T_train_all = train_dataset.T
    propensities_train_all = train_dataset.propensities
    Y_train_all = train_dataset.Y
    Y1_train_all = train_dataset.Y1
    Y0_train_all = train_dataset.Y0
    X_test_all = test_dataset.X
    T_test_all = test_dataset.T
    propensities_test_all = test_dataset.propensities
    Y_test_all = test_dataset.Y
    Y1_test_all = test_dataset.Y1
    Y0_test_all = test_dataset.Y0

    for i, (pretty_loss_name, (loss, loss_bijector, C)) in enumerate(LOSSES.items()):
        print(f"  => {pretty_loss_name}")

        ax = axs[i, j]

        results = []
        for n in tqdm(range(50, 5_000, 200)):
            results.append(run_experiment(n=n, loss=loss, C=C))

        ns = [result.n for result in results]
        if args.what == "outcome":
            ax.plot(
                ns,
                [loss_bijector(result.our_bound_y1) for result in results],
                label="our theoretic bound",
            )
            ax.plot(
                ns,
                [loss_bijector(result.our_empirical_bound_y1) for result in results],
                label="our empirical bound",
            )
            if not args.without_prior_work:
                ax.plot(
                    ns,
                    [loss_bijector(result.prior_work_bound_y1) for result in results],
                    label="prior work",
                )
            ax.plot(
                ns,
                [loss_bijector(result.observable_risk_y1) for result in results],
                alpha=0.2,
                c="k",
                label="observed loss",
            )
            ax.plot(
                ns,
                [loss_bijector(result.causal_risk_y1) for result in results],
                "--",
                c="k",
                label="true complete loss",
            )
        elif args.what == "tlearner":
            ax.plot(
                ns,
                [loss_bijector(result.our_bound_tlearner) for result in results],
                label="our theoretic bound",
            )
            ax.plot(
                ns,
                [
                    loss_bijector(result.our_empirical_bound_tlearner)
                    for result in results
                ],
                label="our empirical bound",
            )
            if not args.without_prior_work:
                ax.plot(
                    ns,
                    [
                        loss_bijector(result.prior_work_bound_tlearner)
                        for result in results
                    ],
                    label="prior work",
                )
            ax.plot(
                ns,
                [loss_bijector(result.observable_risk_y1) for result in results],
                alpha=0.2,
                c="k",
                label="observed loss",
            )
            ax.plot(
                ns,
                [loss_bijector(result.causal_risk_y1) for result in results],
                "--",
                c="k",
                label="true complete loss",
            )
        else:
            raise RuntimeError()
        if not args.without_prior_work:
            ax.set_yscale("log")

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=len(labels),
    frameon=False,
)

print(axs)
plt.subplots_adjust(wspace=0.01, hspace=0.1)
plt.tight_layout()

fig.tight_layout()
fig.savefig(args.output, bbox_inches="tight")
plt.close()
