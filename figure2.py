from argparse import ArgumentParser  # type: ignore

from icecream import ic  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore

from data import load_nearrct, load_observational, load_hiddenconfounding
from metalearners import TLearner


parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset",
    default="hiddenconfounding",
    choices=["nearrct", "observational", "hiddenconfounding"],
)
parser.add_argument("--no-annotations", action="store_true")
parser.add_argument("--no-adjust-viewport", action="store_true")
parser.add_argument("-o", "--output", default="out2.png")
args = parser.parse_args()


rng = np.random.default_rng(0)

if args.dataset == "nearrct":
    data_full = load_nearrct()
elif args.dataset == "observational":
    data_full = load_observational()
elif args.dataset == "hiddenconfounding":
    data_full = load_hiddenconfounding()
else:
    raise ValueError()
data_train, data_test = data_full.train_test_split()
X_train = data_train.X
T_train = data_train.T
propensities_train = data_train.propensities
Y_train = data_train.Y
Y1_train = data_train.Y1
Y0_train = data_train.Y0
X_test = data_test.X
T_test = data_test.T
propensities_test = data_test.propensities
Y_test = data_test.Y
Y1_test = data_test.Y1
Y0_test = data_test.Y0

model = TLearner(base_learner=RandomForestRegressor(random_state=0)).fit(
    X_train, T_train, Y_train
)

prob_t1 = np.mean(T_test == 1)
observational_risk = np.mean(
    (model.model1.predict(X_test[T_test == 1]) - Y1_test[T_test == 1]) ** 2
)
true_risk = np.mean((model.model1.predict(X_test) - Y1_test) ** 2)
chi2 = np.mean((propensities_test / prob_t1 - 1) ** 2)
var = np.var((model.model1.predict(X_test) - Y1_test) ** 2)

margin = lambda lambda_: lambda_ * chi2 + 0.25 * var / lambda_

best_lambda = np.sqrt(0.25 * var / chi2)
lambdas = np.linspace(0.9, 2 * best_lambda, 1000)
margins = margin(lambdas)
ic(best_lambda)

assert np.min(lambdas) <= 1

fig, ax = plt.subplots(1, 1, figsize=(5, 2.8), sharex=True)

ax.set_xlabel(r"$\lambda$")
ax.fill_between(
    lambdas,
    observational_risk - margins,
    observational_risk + margins,
    color="green",
    alpha=0.2,
    label="Bound from Lemma 3.2",
)
ax.plot(
    [best_lambda, best_lambda],
    [
        observational_risk - margin(best_lambda),
        observational_risk + margin(best_lambda),
    ],
    color="k",
    alpha=0.2,
)
ax.axhline(y=true_risk, linestyle="--", color="k", label="True value")
if not args.no_adjust_viewport:
    ax.set_ylim(200, ic(max(observational_risk + margins)) + 1000)
else:
    ax.set_ylim(observational_risk - max(margins), observational_risk + max(margins))
ax.set_yscale("symlog")
ic(observational_risk + margin(best_lambda))
ax.margins(x=0, y=2e5)

if not args.no_annotations:
    ax.annotate(
        "$\\lambda=1$, i.e., no tuning parameter\n(typical generalization bounds)",
        xy=(1 + 0.1, observational_risk + margin(1)),
        xytext=(10, 2e3),
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.2",
            "facecolor": "black",
        },
    )
    ax.annotate(
        "optimal $\\lambda$\n(our bound)",
        xy=(best_lambda, observational_risk + margin(best_lambda)),
        xytext=(best_lambda + 12, 6e2),
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.3",
            "facecolor": "black",
        },
    )

plt.tight_layout()
plt.legend(loc="best")
plt.savefig(args.output, dpi=300, bbox_inches="tight")
plt.close()
