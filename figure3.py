from argparse import ArgumentParser  # type: ignore

from tqdm import tqdm  # type: ignore
from icecream import ic  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import colorzero  # type: ignore
from sklearn.linear_model import Lasso  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from ucimlrepo import fetch_ucirepo  # type: ignore

from metalearners import TLearner, SLearner, XLearner


parser = ArgumentParser()
parser.add_argument("--with-reweighted", action="store_true")
parser.add_argument("--bigger", action="store_true")
parser.add_argument("-o", "--output", default="out3.png")
args = parser.parse_args()


# fetch dataset
parkinsons_telemonitoring = fetch_ucirepo(id=189)

# data (as pandas dataframes)
X_full_raw = parkinsons_telemonitoring.data.features
Y_full_raw = parkinsons_telemonitoring.data.targets

X_full = X_full_raw.drop("sex", axis=1)
X_full["total_UPDRS"] = Y_full_raw["motor_UPDRS"]
X_full = X_full.to_numpy()
T_full = X_full_raw["sex"].to_numpy()
Y_full = Y_full_raw["motor_UPDRS"].to_numpy()

X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X_full, T_full, Y_full, test_size=0.2, random_state=0
)

rng = np.random.default_rng(0)

propensity_scoring_model = CalibratedClassifierCV(
    RandomForestClassifier(random_state=0)
).fit(X_train, T_train)

prob_1_train = np.mean(T_train == 1)
prob_0_train = np.mean(T_train == 0)
MARGIN = 0.1
propensities_train = propensity_scoring_model.predict_proba(X_train)[:, 1]
weights1_test = np.ones(Y_test.shape)
weights0_test = np.ones(Y_test.shape)

C = 2

models_in_main = {
    "Lasso T-learner": TLearner(base_learner=Lasso(random_state=0)),
    "Lasso S-learner": SLearner(base_learner=Lasso(random_state=0)),
    "Lasso X-learner": XLearner(base_learner=Lasso(random_state=0)),
    "Gradient Boosting T-learner": TLearner(
        base_learner=GradientBoostingRegressor(random_state=0)
    ),
    "Gradient Boosting S-learner": SLearner(
        base_learner=GradientBoostingRegressor(random_state=0)
    ),
    "Gradient Boosting X-learner": XLearner(
        base_learner=GradientBoostingRegressor(random_state=0)
    ),
    "Random Forest T-learner": TLearner(
        base_learner=RandomForestRegressor(random_state=0)
    ),
    "Random Forest S-learner": SLearner(
        base_learner=RandomForestRegressor(random_state=0)
    ),
    "Random Forest X-learner": XLearner(
        base_learner=RandomForestRegressor(random_state=0)
    ),
}

data = {}
for og_model_name, model in tqdm(models_in_main.items()):
    for reweighted in [True, False] if args.with_reweighted else [False]:
        model_name = "Reweighted" + og_model_name if reweighted else og_model_name

        if reweighted:
            this_model = model.fit(
                X_train, T_train, Y_train, propensities=propensities_train
            )
        else:
            this_model = model.fit(X_train, T_train, Y_train)

        ALPHA = 0.05

        biased_error_bound_samples = []
        margin_samples = []
        for i in tqdm(range(100)):
            resample = rng.choice(np.arange(len(Y_test)), len(Y_test), replace=True)
            this_X_test = X_test[resample]
            this_T_test = T_test[resample]
            this_Y_test = Y_test[resample]
            this_weights1_test = weights1_test[resample]
            this_weights0_test = weights0_test[resample]

            prob_1 = np.mean(this_T_test == 1)
            prob_0 = np.mean(this_T_test == 0)
            chi2_term_1 = 2 * (
                np.mean(
                    this_weights1_test**2
                    * (
                        propensity_scoring_model.predict_proba(this_X_test)[:, 1]
                        - this_T_test
                    )
                    ** 2
                )
                / prob_1**2
                + np.mean((this_weights1_test * this_T_test / prob_1 - 1) ** 2)
            )
            chi2_term_0 = 2 * (
                np.mean(
                    this_weights0_test**2
                    * (
                        propensity_scoring_model.predict_proba(this_X_test)[:, 0]
                        - (1 - this_T_test)
                    )
                    ** 2
                )
                / prob_0**2
                + np.mean((this_weights0_test * (1 - this_T_test) / prob_0 - 1) ** 2)
            )

            if isinstance(model, TLearner) or isinstance(model, SLearner):
                var_term = 1e-5

                if isinstance(model, TLearner):
                    preds1 = this_model.model1.predict(
                        this_X_test[this_T_test == 1, ...]
                    )
                    preds0 = this_model.model0.predict(
                        this_X_test[this_T_test == 0, ...]
                    )
                elif isinstance(model, SLearner):
                    preds1 = this_model.individual_predict(
                        this_X_test[this_T_test == 1, ...], t=1
                    )
                    preds0 = this_model.individual_predict(
                        this_X_test[this_T_test == 0, ...], t=0
                    )
                else:
                    raise TypeError()

                mse_test_1 = np.mean(
                    this_weights1_test[this_T_test == 1]
                    * (preds1 - this_Y_test[this_T_test == 1]) ** 2
                )
                mse_test_0 = np.mean(
                    this_weights0_test[this_T_test == 0]
                    * (preds0 - this_Y_test[this_T_test == 0]) ** 2
                )

                best_lambda_1 = np.sqrt(0.25 * var_term / chi2_term_1)
                best_lambda_0 = np.sqrt(0.25 * var_term / chi2_term_0)

                margin_1 = best_lambda_1 * chi2_term_1 + 0.25 * var_term / best_lambda_1
                margin_0 = best_lambda_0 * chi2_term_0 + 0.25 * var_term / best_lambda_0
                bound_1 = mse_test_1 + margin_1
                bound_0 = mse_test_0 + margin_0
                biased_bound_cate = C * (mse_test_1 + mse_test_0)
                margin = C * (margin_1 + margin_0)
            elif isinstance(model, XLearner):
                var_term = 1e-5

                preds1 = this_model.model1.predict(this_X_test[this_T_test == 1, ...])
                preds0 = this_model.model0.predict(this_X_test[this_T_test == 0, ...])
                preds10 = this_model.model10.predict(this_X_test[this_T_test == 1, ...])
                preds01 = this_model.model01.predict(this_X_test[this_T_test == 0, ...])
                e1 = this_model.propensity_model.predict_proba(this_X_test)[:, 1]
                e0 = this_model.propensity_model.predict_proba(this_X_test)[:, 0]

                mse_test_1 = np.mean(
                    this_weights1_test[this_T_test == 1]
                    * (e0[this_T_test == 1] * (preds1 - this_Y_test[this_T_test == 1]))
                    ** 2
                )
                mse_test_0 = np.mean(
                    this_weights0_test[this_T_test == 0]
                    * (e1[this_T_test == 0] * (preds0 - this_Y_test[this_T_test == 0]))
                    ** 2
                )
                mse_test_10 = np.mean(
                    this_weights1_test[this_T_test == 1]
                    * (
                        e1[this_T_test == 1]
                        * (
                            preds10
                            - (
                                this_Y_test[this_T_test == 1]
                                - this_model.model0.predict(
                                    this_X_test[this_T_test == 1, ...]
                                )
                            )
                        )
                    )
                    ** 2
                )
                mse_test_01 = np.mean(
                    this_weights0_test[this_T_test == 0]
                    * (
                        e0[this_T_test == 0]
                        * (
                            preds01
                            - (
                                this_model.model1.predict(
                                    this_X_test[this_T_test == 0, ...]
                                )
                                - this_Y_test[this_T_test == 0]
                            )
                        )
                    )
                    ** 2
                )

                best_lambda_1 = np.sqrt(0.25 * var_term / chi2_term_1)
                best_lambda_0 = np.sqrt(0.25 * var_term / chi2_term_0)

                margin_1 = best_lambda_1 * chi2_term_1 + 0.25 * var_term / best_lambda_1
                margin_0 = best_lambda_0 * chi2_term_0 + 0.25 * var_term / best_lambda_0
                bound_1 = mse_test_1 + 2 * margin_1
                bound_0 = mse_test_0 + 2 * margin_0
                biased_bound_cate = (
                    C * C * (mse_test_1 + mse_test_0 + mse_test_10 + mse_test_01)
                )
                margin = C * C * (margin_1 + margin_0)
            else:
                raise TypeError()

            biased_error_bound_samples.append(biased_bound_cate)
            margin_samples.append(margin)

        data[model_name] = {
            "biased_error_bound": (
                np.quantile(biased_error_bound_samples, ALPHA),
                np.median(biased_error_bound_samples),
                np.quantile(biased_error_bound_samples, 1 - ALPHA),
            ),
            "margin": (
                np.quantile(margin_samples, ALPHA),
                np.median(margin_samples),
                np.quantile(margin_samples, 1 - ALPHA),
            ),
        }

ic(data)

if args.bigger:
    fig, ax = plt.subplots(figsize=(12, 7))
else:
    fig, ax = plt.subplots(figsize=(12, 3))

colors = {
    "T-learner": colorzero.Color("#469990"),  # teal
    "S-learner": colorzero.Color("#3cb44b"),  # green
    "X-learner": colorzero.Color("#4363d8"),  # blue
}

for i, (model_name, model_results) in enumerate(data.items()):
    color = colors[model_name.split(" ")[-1]]

    ax.plot(
        [
            max(model_results["biased_error_bound"][0] - model_results["margin"][2], 0),
            max(model_results["biased_error_bound"][2] + model_results["margin"][0], 0),
        ],
        [i, i],
        linewidth=8,
        color=(color + colorzero.Lightness(0.2)).html,
    )
    knob_start = model_results["biased_error_bound"][0]
    ax.plot(
        [
            max(knob_start, 0),
            max(max(knob_start + 0.0001, model_results["biased_error_bound"][2]), 0),
        ],
        [i, i],
        linewidth=12,
        color=(color + colorzero.Lightness(0.1)).html,
    )
margin = 0.5
ax.set_ylim(0 - margin, len(data) - 1 + margin)
ax.set_yticks(np.arange(len(data)), list(data.keys()))

fig.tight_layout()
fig.savefig(args.output, bbox_inches="tight")
