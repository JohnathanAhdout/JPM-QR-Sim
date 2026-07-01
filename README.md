# JPM QR Simulation: Quantitative Modeling Portfolio

This repository consolidates a set of quantitative finance and credit-risk modeling exercises completed as part of a JPMorgan Chase quantitative research simulation. The work spans both structured credit-risk analysis and time-series commodity modeling, with an emphasis on interpretable statistical methods, model validation, and practical risk applications.

## Project Summary

Across these scripts, I implemented and validated several core modeling workflows:

- Built a logistic-regression framework to estimate borrower probability of default (PD) from loan-level characteristics such as debt-to-income, payment-to-income, years employed, credit lines outstanding, and FICO score.
- Evaluated model performance using stratified cross-validation and ROC-AUC to assess out-of-sample discrimination quality.
- Extended the credit-risk workflow with a FICO score quantization approach that uses dynamic programming and a log-likelihood objective to create categorical rating bands from continuous credit scores, enabling compatibility with categorical-model architectures.
- Estimated expected loss (EL) using the standard credit-risk formulation $EL = PD \times EAD \times (1 - RR)$, using a recovery-rate assumption to translate default probabilities into loss estimates.
- Developed a time-series natural-gas pricing workflow that loads historical price data, fits a trend-plus-seasonal model, and uses sine/cosine terms to capture recurring seasonal structure.
- Implemented a storage-contract valuation prototype that models inventory evolution, purchase and withdrawal events, storage costs, and fees to estimate the economic value of a natural-gas storage position.

## Files

- `loan_default_model.py`: credit-risk modeling workflow including PD estimation, expected-loss calculation, FICO bucketing, and validation metrics.
- `natural_gas_estimator.py`: natural-gas price estimation and storage-contract valuation using a seasonality-aware forecasting approach.
- `Task 3 and 4_Loan_Data.csv`: historical loan-book data used for default-model training and evaluation.
- `Nat_Gas.csv`: monthly natural-gas price data used for price estimation and forecasting.

## Technical Approach

The credit-risk component uses an interpretable supervised-learning framework in which historical borrower data is transformed into compact features, then used to fit a logistic-regression classifier. The implementation emphasizes transparency and reproducibility: feature engineering is explicit, cross-validation is used for robustness, and the output includes calibrated default probabilities and expected-loss metrics for a given loan profile. The FICO bucketing component represents a separate quantization task in which the continuous score distribution is partitioned into discrete intervals that best separate default outcomes according to a likelihood-based objective.

The natural-gas component follows a similar philosophy of transparent modeling. It uses historical prices to estimate a baseline trend and seasonal structure, then applies the fitted structure to infer historical values and extrapolate forward. The storage-contract script then evaluates a hypothetical storage strategy by processing transactions chronologically and tracking inventory, costs, and fees.

## How to Run

```bash
python loan_default_model.py
```

To run the natural-gas workflow:

```bash
python natural_gas_estimator.py
```

## Purpose for the JPM QR Simulation

This repository was created as part of a JPMorgan Chase quantitative research simulation to demonstrate practical experience in quantitative modeling, risk analytics, and data-driven decision-making. It reflects a structured approach to building interpretable models for default prediction, credit-risk estimation, and time-series forecasting in a finance context.
