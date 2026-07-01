# Credit-Risk Modeling Prototype

This project implements a simple credit-risk workflow for loan and mortgage portfolios. It includes:

- a logistic-regression model for estimating borrower probability of default (PD)
- a FICO-score bucketing approach that converts continuous FICO values into categorical rating bands
- expected-loss estimation using PD, exposure at default, and an assumed recovery rate

## Files

- `loan_default_model.py`: main script containing the full workflow
- `Task 3 and 4_Loan_Data.csv`: historical loan data used for modeling
- `Nat_Gas.csv` and `natural_gas_estimator.py`: additional unrelated analysis files from the workspace

## How to run

```bash
python loan_default_model.py
```

## What the model does

1. Loads historical loan data.
2. Builds interpretable borrower features such as debt-to-income and payment-to-income ratios.
3. Trains a logistic-regression model to estimate default probability.
4. Uses dynamic programming and log-likelihood optimization to bucket FICO scores into categorical ratings.
5. Estimates expected loss for a sample borrower profile.

## Notes

This is a prototype and educational project intended for demonstration and portfolio-risk analysis.
