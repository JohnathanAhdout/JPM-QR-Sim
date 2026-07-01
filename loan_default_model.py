from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_FILE = Path(__file__).with_name("Task 3 and 4_Loan_Data.csv")

# This script treats the loan book as a supervised classification problem in which the model learns from historical borrower characteristics and tries to predict whether a borrower will default.
# We read the historical loan data, create a compact set of features, fit a logistic-regression model, and then use that model to estimate the probability of default.
# Once a borrower-level PD has been estimated, the expected loss is computed as exposure at default multiplied by the PD and by one minus the assumed recovery rate, which is the standard credit-risk formulation.


def _load_data() -> pd.DataFrame:
    """Load the loan book and keep the core borrower features that the model will use."""
    data = pd.read_csv(DATA_FILE)  # read the CSV file that contains the historical loan-book data and the default outcome flag
    return data  # return the full table so the feature engineering step can build the compact modeling matrix


DATA = _load_data()
TARGET_COLUMN = "default"


def _log_likelihood_for_bin(defaults_in_bin: int, total_in_bin: int) -> float:
    """Compute the bin-level log-likelihood for a bucket that contains a share of defaults."""
    if total_in_bin <= 0:
        return 0.0  # if a bucket has no observations, it contributes nothing to the objective

    default_rate = defaults_in_bin / total_in_bin  # estimate the default probability inside the bucket from the data itself
    default_rate = min(max(default_rate, 1e-12), 1.0 - 1e-12)  # clip the rate so the log-likelihood stays numerically stable
    return defaults_in_bin * math.log(default_rate) + (total_in_bin - defaults_in_bin) * math.log(1.0 - default_rate)  # reward partitions that separate default-heavy and default-light regions cleanly


def fit_fico_rating_map(scores: pd.Series, defaults: pd.Series, n_buckets: int = 10) -> tuple[list[float], pd.Series]:
    """Learn score boundaries that partition FICO values into a fixed number of buckets using a log-likelihood objective."""
    if n_buckets <= 1:
        raise ValueError("n_buckets must be at least 2.")  # a single bucket would not help the model distinguish different credit quality bands
    if len(scores) != len(defaults):
        raise ValueError("scores and defaults must contain the same number of observations.")  # the optimization needs one outcome per score to evaluate each candidate bucket

    working_frame = pd.DataFrame({"fico_score": scores, "default": defaults}).dropna().copy()  # keep only rows with usable values so the optimization is not distorted by missing data
    working_frame = working_frame.sort_values("fico_score").reset_index(drop=True)  # sort the data so the optimization can work from low score to high score

    aggregated = (  # aggregate identical scores together so the dynamic-programming step is efficient and still faithful to the full dataset
        working_frame.groupby("fico_score", as_index=False)
        .agg(records=("default", "size"), defaults=("default", "sum"))
        .sort_values("fico_score")
        .reset_index(drop=True)
    )

    unique_scores = aggregated["fico_score"].to_numpy(dtype=float)  # these are the candidate score values at which the bucket boundaries can sit
    records = aggregated["records"].to_numpy(dtype=int)  # each distinct score can carry many borrowers, so we preserve the population size for each level
    defaults = aggregated["defaults"].to_numpy(dtype=int)  # each distinct score also contributes its observed number of defaults to the likelihood calculation

    prefix_records = np.zeros(len(unique_scores) + 1, dtype=int)  # cumulative record counts let us query any interval in constant time
    prefix_defaults = np.zeros(len(unique_scores) + 1, dtype=int)  # cumulative default counts do the same for the default outcome
    prefix_records[1:] = np.cumsum(records)
    prefix_defaults[1:] = np.cumsum(defaults)

    dp = np.full((n_buckets + 1, len(unique_scores) + 1), -np.inf)  # dp[bucket_index, end_position] stores the best objective value for the first end_position score levels using bucket_index buckets
    parent = np.full((n_buckets + 1, len(unique_scores) + 1), -1, dtype=int)  # parent pointers let us reconstruct the chosen cut points after the dynamic program finishes
    dp[0, 0] = 0.0  # with zero buckets, the objective is zero for an empty prefix

    for bucket_index in range(1, n_buckets + 1):  # build the partition bucket by bucket from the bottom of the score range upward
        for end_position in range(bucket_index, len(unique_scores) + 1):  # the current bucket must cover at least one unique score level
            best_value = -np.inf
            best_split = -1

            for start_position in range(bucket_index - 1, end_position):  # the previous partition must leave at least one level for the current bucket to cover
                records_in_bin = prefix_records[end_position] - prefix_records[start_position]  # count all borrowers in the candidate interval
                defaults_in_bin = prefix_defaults[end_position] - prefix_defaults[start_position]  # count defaulted borrowers in the same interval
                candidate_value = dp[bucket_index - 1, start_position] + _log_likelihood_for_bin(defaults_in_bin, records_in_bin)  # score the candidate interval under the chosen objective

                if candidate_value > best_value:  # keep the split that improves the total likelihood the most
                    best_value = candidate_value
                    best_split = start_position

            dp[bucket_index, end_position] = best_value  # store the best objective for this many buckets up to this end point
            parent[bucket_index, end_position] = best_split  # remember where the last cut was applied

    split_positions = []  # we will reconstruct the boundary positions from the parent pointers in reverse order
    current_bucket = n_buckets
    current_end = len(unique_scores)
    while current_bucket > 1:  # walk backward until the first bucket is reconstructed
        split_position = parent[current_bucket, current_end]
        if split_position < 0:
            raise ValueError("The dynamic-programming search failed to construct a valid partition.")  # a valid solution should always produce a non-negative split pointer
        split_positions.append(split_position)
        current_end = split_position
        current_bucket -= 1

    split_positions.reverse()  # the parent pointers were recovered from the back of the partition, so the list must be flipped for forward order
    boundaries = [float(unique_scores[position]) for position in split_positions]  # each boundary is the first score value in the later bucket, which is what future assignment logic will use

    rating_series = apply_fico_rating_map(scores, boundaries)  # apply the learned boundaries to the original score series so the output is directly usable
    return boundaries, rating_series


def apply_fico_rating_map(scores: pd.Series, boundaries: list[float]) -> pd.Series:
    """Apply a learned set of boundaries to a score series and return a discrete rating where lower values are better."""
    score_series = pd.Series(scores).copy()  # work on a copy so the caller's data is not modified during the assignment step
    rating_series = pd.Series(np.nan, index=score_series.index, dtype=float)  # missing values stay missing rather than being forced into an arbitrary bucket

    valid_mask = score_series.notna()  # only real scores should be assigned a rating
    if valid_mask.sum() == 0:
        return rating_series.astype(int)  # if nothing is present, return an empty integer series

    score_values = score_series.loc[valid_mask].to_numpy(dtype=float)
    bucket_ids = np.digitize(score_values, boundaries, right=False)  # lower scores fall into earlier buckets and higher scores fall into later buckets
    ratings = (len(boundaries) + 1) - bucket_ids  # reverse the bucket order so a lower rating means stronger credit quality
    rating_series.loc[valid_mask] = ratings
    return rating_series.astype(int)


def summarize_fico_rating_map(scores: pd.Series, defaults: pd.Series, n_buckets: int = 10) -> pd.DataFrame:
    """Create a compact summary table that shows how many borrowers and how many defaults sit in each learned rating bucket."""
    boundaries, ratings = fit_fico_rating_map(scores, defaults, n_buckets=n_buckets)  # first learn the boundaries from the historical data
    summary_frame = pd.DataFrame({"fico_score": scores, "rating": ratings, "default": defaults}).dropna()  # then join the ratings back to the observed outcomes for reporting
    summary_table = (  # aggregate the borrowers and defaults within each rating band so the risk team can inspect the distribution quickly
        summary_frame.groupby("rating")
        .agg(
            borrowers=("default", "size"),
            defaults=("default", "sum"),
            default_rate=("default", "mean"),
            min_fico=("fico_score", "min"),
            max_fico=("fico_score", "max"),
        )
        .sort_index()
    )
    summary_table.index.name = "rating"
    return summary_table, boundaries


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a compact feature matrix that mirrors the example answer while keeping the modeling process interpretable."""
    frame = frame.copy()  # work on a copy so the original data is preserved while the ratios are derived
    frame["payment_to_income"] = frame["loan_amt_outstanding"] / frame["income"].clip(lower=1.0)  # create the payment-to-income ratio that the example answer uses to capture affordability
    frame["debt_to_income"] = frame["total_debt_outstanding"] / frame["income"].clip(lower=1.0)  # create the debt-to-income ratio that highlights balance-sheet pressure
    return frame[[  # keep only the simple, interpretable features that will feed the logistic-regression model
        "credit_lines_outstanding",
        "debt_to_income",
        "payment_to_income",
        "years_employed",
        "fico_score",
    ]]


def _fit_model() -> tuple[LogisticRegression, dict[str, float | list[float]]]:
    """Fit a simple logistic-regression model and evaluate it with robust cross-validation."""
    features = _prepare_features(DATA)  # build the compact feature matrix from the historical loan book
    target = DATA[TARGET_COLUMN]  # pull out the default indicator, which is the target variable the model is trying to predict

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # split the data into several folds so the validation is more stable than a single train/test split
    scores = cross_val_score(LogisticRegression(random_state=0, solver="liblinear", tol=1e-5, max_iter=10000), features, target, cv=cv, scoring="roc_auc")  # estimate the model's out-of-sample ROC-AUC across multiple folds

    model = LogisticRegression(random_state=0, solver="liblinear", tol=1e-5, max_iter=10000)  # create the final model that will be used for scoring new borrower profiles
    model.fit(features, target)  # fit the model on the full dataset once its performance has been validated

    return model, {  # store the fitted model along with the validation statistics so the script can report them clearly
        "roc_auc": float(scores.mean()),
        "roc_auc_std": float(scores.std()),
        "fold_scores": [float(score) for score in scores],
    }


MODEL, MODEL_VALIDATION = _fit_model()


def predict_default_probability(profile: dict[str, float | int]) -> float:
    """Estimate the probability of default for a borrower profile using the fitted logistic-regression model."""
    prepared = {  # turn the incoming borrower attributes into the same field names used during training
        "credit_lines_outstanding": float(profile.get("credit_lines_outstanding", 0)),
        "loan_amt_outstanding": float(profile.get("loan_amt_outstanding", 0)),
        "total_debt_outstanding": float(profile.get("total_debt_outstanding", 0)),
        "income": float(profile.get("income", 1)),
        "years_employed": float(profile.get("years_employed", 0)),
        "fico_score": float(profile.get("fico_score", 300)),
    }

    transformed = pd.DataFrame([prepared])  # wrap the profile in a one-row DataFrame so the same feature engineering logic can be applied
    transformed["payment_to_income"] = transformed["loan_amt_outstanding"] / transformed["income"].clip(lower=1.0)  # compute the affordability ratio using the same logic as the example answer
    transformed["debt_to_income"] = transformed["total_debt_outstanding"] / transformed["income"].clip(lower=1.0)  # compute the balance-sheet pressure ratio

    features = transformed[[  # keep the same compact feature layout that was used during training
        "credit_lines_outstanding",
        "debt_to_income",
        "payment_to_income",
        "years_employed",
        "fico_score",
    ]]

    return float(MODEL.predict_proba(features)[0, 1])  # ask the fitted logistic-regression model for the probability that this borrower defaults


def estimate_expected_loss(profile: dict[str, float | int], recovery_rate: float = 0.10) -> float:
    """Estimate expected loss for a loan using the probability of default and a recovery-rate assumption."""
    pd_value = predict_default_probability(profile)  # first estimate the probability that the borrower will default
    exposure = float(profile.get("loan_amt_outstanding", 0.0))  # use the outstanding loan balance as the exposure at default
    return float(pd_value * exposure * (1.0 - recovery_rate))  # apply the standard expected-loss formula with a 10% recovery assumption


def compare_models() -> dict[str, float | list[float]]:
    """Return the out-of-sample validation statistics for the interpretable logistic-regression model."""
    return {  # report the validation summary for the simple model so the user can see how stable the fit is
        "roc_auc": MODEL_VALIDATION["roc_auc"],
        "roc_auc_std": MODEL_VALIDATION["roc_auc_std"],
        "fold_scores": MODEL_VALIDATION["fold_scores"],
    }


def run_self_tests() -> None:
    """Run a compact set of self-checks to confirm the default-probability workflow behaves sensibly."""
    sample_profile = {  # define a representative borrower profile that exercises the prediction pipeline with plausible values
        "credit_lines_outstanding": 2,
        "loan_amt_outstanding": 5000.0,
        "total_debt_outstanding": 8000.0,
        "income": 60000.0,
        "years_employed": 3,
        "fico_score": 650,
    }

    default_probability = predict_default_probability(sample_profile)  # estimate the PD for a representative borrower profile
    expected_loss = estimate_expected_loss(sample_profile)  # convert the PD into an expected-loss number using the 10% recovery assumption
    comparison = compare_models()  # inspect how the baseline and the more advanced model compare on out-of-sample AUC

    assert 0.0 <= default_probability <= 1.0  # a default probability should always lie between zero and one because it is a calibrated probability output
    assert math.isfinite(expected_loss) and expected_loss >= 0.0  # expected loss should be a finite, non-negative amount because it is a monetary quantity derived from a probability
    assert comparison["roc_auc"] >= 0.0  # the model's average AUC should be a valid score across the validation folds


if __name__ == "__main__":
    run_self_tests()  # run the built-in checks before the script produces any user-facing results
    print("Self-tests passed.")  # print a short message confirming that the internal checks succeeded

    sample_profile = {
        "credit_lines_outstanding": 2,
        "loan_amt_outstanding": 5000.0,
        "total_debt_outstanding": 8000.0,
        "income": 60000.0,
        "years_employed": 3,
        "fico_score": 650,
    }

    print("Default probability:", predict_default_probability(sample_profile))  # print the estimated PD for the sample borrower
    print("Expected loss:", estimate_expected_loss(sample_profile))  # print the corresponding expected loss under a 10% recovery rate

    fico_summary, fico_boundaries = summarize_fico_rating_map(DATA["fico_score"], DATA[TARGET_COLUMN], n_buckets=10)  # learn a general FICO rating map from the historical data and summarize the resulting buckets
    print("FICO rating boundaries:", fico_boundaries)  # show the score thresholds that define each credit-quality band
    print("FICO rating summary:\n", fico_summary)  # display how many borrowers and defaults fall into each learned rating bucket
    print("Model comparison:", compare_models())  # print the out-of-sample AUC comparison for both models
