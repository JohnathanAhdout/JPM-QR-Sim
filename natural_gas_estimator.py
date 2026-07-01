from __future__ import annotations

import csv  # import the standard-library CSV module so the script can read the monthly snapshot file row by row
from datetime import date, datetime, timedelta  # import date helpers so the script can work with calendar dates and simple time shifts
from pathlib import Path  # import Path so the script can find the CSV file relative to itself
from typing import Sequence  # import Sequence so the new pricing function can accept lists of dates and prices cleanly

import numpy as np  # import NumPy because it lets the script do fast vectorized regression and interpolation operations on the price series

DATA_FILE = Path(__file__).with_name("Nat_Gas.csv")  # locate the CSV file next to this script so the model always reads the same data source

# This script uses a lightweight econometric-style baseline because the dataset is short and monthly, which makes a simple trend-and-seasonality model easy to fit and interpret.
# For a production-grade commodity forecast, more advanced options would include ARIMA/SARIMA for linear seasonal structure, random forests or XGBoost for non-linear patterns, and LSTM models for long-range dependence.


def _load_data() -> tuple[list[date], list[float]]:  # define a loader that returns two parallel lists: one for dates and one for prices
    dates: list[date] = []  # create an empty list that will hold each date as a Python date object
    prices: list[float] = []  # create an empty list that will hold each price as a floating-point number
    with DATA_FILE.open(newline="", encoding="utf-8") as handle:  # open the CSV file in text mode with UTF-8 encoding so the data can be parsed reliably
        reader = csv.DictReader(handle)  # create a CSV reader that reads each row as a dictionary keyed by the column names
        for row in reader:  # loop through the rows one by one so each monthly observation is processed independently
            dt = datetime.strptime(row["Dates"], "%m/%d/%y").date()  # parse the text date into a real Python date object using the format found in the CSV
            price = float(row["Prices"].replace("E+", "e+"))  # convert the scientific-notation text into a numeric float so the model can do arithmetic on it
            dates.append(dt)  # append the parsed date to the dates list so the time series remains aligned with the price observations
            prices.append(price)  # append the parsed price to the prices list so the regression model can use the numeric values later
    return dates, prices  # return the two lists so the rest of the script can use them as the observed history


DATES, PRICES = _load_data()  # run the loader once at import time so the script has the full historical series available immediately
PRICES_ARRAY = np.array(PRICES, dtype=float)  # turn the price list into a NumPy array so math operations can be vectorized and faster
TIME_INDEX = np.arange(len(DATES), dtype=float)  # create a numeric index 0, 1, 2, ... that represents the order of each monthly observation in the time series
TIME_MEAN = float(TIME_INDEX.mean())  # compute the average time index and center the regression around it so the intercept remains numerically stable


def _fit_trend_and_seasonal_model() -> tuple[float, float, float, float, float, float]:  # define a simple trend-and-seasonality model that mirrors the JPMC example by combining linear drift with a sine-cosine seasonal component
    """Fit a linear trend and an annual sine-cosine seasonal pattern to the observed price series."""
    start_date = DATES[0]  # use the first observation as the origin
    days_from_start = np.array([(d - start_date).days for d in DATES], dtype=float)  # convert each observation date into a day count from the initial sample date

    x_mean = float(days_from_start.mean())  # compute the average time value so the regression slope and intercept are centered neatly
    y_mean = float(PRICES_ARRAY.mean())  # compute the average price, which is the intercept term in a simple linear regression
    trend_slope = float(np.sum((days_from_start - x_mean) * (PRICES_ARRAY - y_mean)) / np.sum((days_from_start - x_mean) ** 2))  # estimate the linear trend using a standard regression formula
    trend_intercept = float(y_mean - trend_slope * x_mean)  # recover the intercept so the trend line can be evaluated at any point in time

    detrended = PRICES_ARRAY - (trend_slope * days_from_start + trend_intercept)  # remove the linear trend so the remaining behavior can be attributed to seasonal movement
    annual_period = 365.0  # assume the recurring seasonal cycle is roughly one year
    sin_term = np.sin(days_from_start * 2.0 * np.pi / annual_period)  # create the sine-seasonality regressor
    cos_term = np.cos(days_from_start * 2.0 * np.pi / annual_period)  # create the cosine-seasonality regressor

    sine_coef = float(np.sum(detrended * sin_term) / np.sum(sin_term**2)) if np.sum(sin_term**2) > 0 else 0.0  # fit the sine coefficient using the detrended series
    cosine_coef = float(np.sum(detrended * cos_term) / np.sum(cos_term**2)) if np.sum(cos_term**2) > 0 else 0.0  # fit the cosine coefficient using the detrended series

    fitted = trend_slope * days_from_start + trend_intercept + sine_coef * sin_term + cosine_coef * cos_term  # build the full fitted line, including the seasonal component
    residuals = PRICES_ARRAY - fitted  # compute the unexplained residuals after the trend and seasonality are removed

    if len(residuals) > 2:  # only compute a persistence term when there are enough residuals to estimate short-run dependence reliably
        lagged = residuals[1:]  # create the later residuals so each observation can be compared with the one before it
        previous = residuals[:-1]  # create the earlier residuals that precede each later residual
        corr = float(np.corrcoef(previous, lagged)[0, 1]) if len(previous) > 1 else 0.0  # compute the correlation between consecutive residuals, which is the simplest AR(1)-style persistence measure
        ar1 = corr if np.isfinite(corr) else 0.0  # keep the persistence value only if it is finite; otherwise set it to zero to avoid invalid values
    else:  # if the series is too short, do not invent persistence from too little evidence
        ar1 = 0.0  # set the persistence term to zero because there is no robust short-run dependence estimate available

    return trend_slope, trend_intercept, sine_coef, cosine_coef, ar1, float(residuals[-1])  # return the fitted trend, seasonal coefficients, persistence, and the last residual for the forecast


TREND_SLOPE, TREND_INTERCEPT, SEASONAL_SINE_COEF, SEASONAL_COSINE_COEF, AR1_COEFFICIENT, LAST_RESIDUAL = _fit_trend_and_seasonal_model()  # fit the improved model once at import time so the forecast is based on the full observed history


def _predict_structure(target_date: date) -> float:  # define a helper that computes the trend-and-seasonal structure of the forecast before the persistence adjustment
    start_date = DATES[0]  # use the same origin date that was used when the model was fitted
    days_from_start = (target_date - start_date).days  # convert the target date into the same day-count scale as the fitted regression
    annual_period = 365.0  # keep the seasonal cycle at roughly one year, which is the natural period
    seasonal_component = SEASONAL_SINE_COEF * np.sin(days_from_start * 2.0 * np.pi / annual_period) + SEASONAL_COSINE_COEF * np.cos(days_from_start * 2.0 * np.pi / annual_period)  # add the sine-cosine seasonal effect to the linear trend
    return float(TREND_INTERCEPT + TREND_SLOPE * days_from_start + seasonal_component)  # return the structural forecast component before the persistence correction is added


def _month_index(target_date: date) -> float:  # define a helper that turns a calendar date into a monthly index so interpolation and forecasting use the same time unit
    return (target_date.year - DATES[0].year) * 12 + (target_date.month - DATES[0].month)  # compute the number of months between the first observation and the target date, which is equivalent to a monthly time index


def estimate_price(target_date: date) -> float:  # define the main function that returns a price estimate for any requested date
    """Estimate natural gas price by interpolating within the observed monthly sample and using an econometric-style forecast beyond it."""  # explain that the function uses historical interpolation for observed dates and a structured forecast for future dates
    if target_date < DATES[0]:  # reject any date earlier than the first available observation because the model has no historical information for it
        raise ValueError("Requested date is earlier than available data")  # raise a clear error so the caller knows the request is outside the supported range

    if target_date <= DATES[-1]:  # if the requested date is within the observed sample, use interpolation rather than extrapolation
        if target_date in DATES:  # if the date is exactly one of the monthly snapshot dates, return the historical price directly to avoid any approximation
            idx = DATES.index(target_date)  # find the position of the exact date in the list of dates
            return PRICES[idx]  # return the historical price at that exact position

        known_indices = np.array([_month_index(d) for d in DATES], dtype=float)  # create an array of monthly indices for all observed dates so interpolation can happen on a numeric axis
        target_index = _month_index(target_date)  # convert the requested date into the same monthly index scale as the historical observations
        return float(np.interp(target_index, known_indices, PRICES_ARRAY))  # interpolate linearly between the nearest monthly snapshots to estimate the price at the requested date

    months_since_last = _month_index(target_date) - _month_index(DATES[-1])  # count how many months lie between the last observed month and the requested future month
    if months_since_last <= 0:  # guard against a future date that somehow ends up on or before the last observation
        return float(PRICES[-1])  # return the latest observed price if the date is not actually beyond the sample

    regression_value = _predict_structure(target_date)  # compute the structural trend-and-seasonal component for the target month using the fitted sine-cosine seasonal model
    persistence_adjustment = AR1_COEFFICIENT * LAST_RESIDUAL  # apply a small persistence correction based on the final residual, which mimics an AR(1) step in an econometric model
    return float(regression_value + persistence_adjustment)  # add the structural forecast and persistence correction to obtain the final estimate


def price_storage_contract(
    injection_dates: Sequence[date],
    withdrawal_dates: Sequence[date],
    injection_prices: Sequence[float] | None = None,
    withdrawal_prices: Sequence[float] | None = None,
    injection_rate: float = 1_000_000.0,
    withdrawal_rate: float = 1_000_000.0,
    max_storage: float = float("inf"),
    storage_cost_per_month: float = 0.0,
    injection_cost_per_unit: float = 0.0,
    withdrawal_cost_per_unit: float = 0.0,
    return_details: bool = False,
) -> float | dict[str, object]:  # define a function that values a storage contract by simulating buy and sell cash flows over time
    """Value a natural-gas storage contract by combining purchase, sale, and inventory-cost cash flows."""
    if injection_prices is not None and len(injection_dates) != len(injection_prices):  # check that the caller supplied a price for each injection date if prices were provided
        raise ValueError("Injection prices must match the number of injection dates")  # raise a clear error if the input lists are inconsistent
    if withdrawal_prices is not None and len(withdrawal_dates) != len(withdrawal_prices):  # check that the caller supplied a price for each withdrawal date if prices were provided
        raise ValueError("Withdrawal prices must match the number of withdrawal dates")  # raise a clear error if the input lists are inconsistent
    if injection_rate <= 0 or withdrawal_rate <= 0:  # reject non-positive rates because a storage contract with zero or negative flow would not make economic sense
        raise ValueError("Injection and withdrawal rates must be positive")  # raise a clear error so the caller knows the input is invalid

    all_dates = sorted(set(injection_dates + withdrawal_dates))  # make the event ordering explicit, so the contract is processed in chronological order

    inventory = 0.0  # track the amount of gas currently stored so the model can enforce the maximum storage limit
    buy_cost = 0.0  # sum the purchase cost of gas and any injection fee incurred by the holder
    cash_in = 0.0  # sum the sale proceeds received from withdrawals and subtract any withdrawal fee
    storage_cost = 0.0  # sum the carrying cost incurred while gas sits in storage between events
    breakdown: list[dict[str, object]] = []  # keep a detailed timeline so the user can see exactly how the contract value was built
    previous_date: date | None = None  # remember the previous event date so storage costs can be charged over the interval between events

    for event_date in all_dates:  # loop over the ordered calendar dates and process injections and withdrawals 
        is_injection = event_date in injection_dates  # check whether the current date is an injection date
        is_withdrawal = event_date in withdrawal_dates  # check whether the current date is a withdrawal date

        if previous_date is not None and event_date > previous_date and inventory > 0:  # charge storage cost only when the contract holds inventory across a non-zero interval
            months = (event_date.year - previous_date.year) * 12 + (event_date.month - previous_date.month)  # convert the gap between events into a monthly duration for storage-cost charging
            if months > 0:  # only charge storage cost if the interval spans at least one month
                carrying_cost = storage_cost_per_month * inventory * months  # compute the storage bill as cost per month times the inventory carried for that many months
                storage_cost += carrying_cost  # add the carrying cost to the running storage bill
                breakdown.append(
                    {
                        "date": event_date,
                        "type": "storage_cost",
                        "volume": inventory,
                        "price": None,
                        "cash_flow": -carrying_cost,
                        "inventory_before": inventory,
                        "inventory_after": inventory,
                    }
                )

        if is_injection:  # if the event is an injection, the holder buys gas and adds it to inventory
            idx = injection_dates.index(event_date)  # find the matching injection date so the correct price can be attached to the event
            price = injection_prices[idx] if injection_prices is not None else estimate_price(event_date)  # use the supplied price if provided, otherwise estimate it with the model
            injected_volume = min(injection_rate, max_storage - inventory)  # only buy as much as the storage facility can hold, respecting the maximum storage constraint
            if injected_volume > 0:
                purchase_cost = injected_volume * (price + injection_cost_per_unit)  # compute the gross purchase payment including any injection fee
                buy_cost += purchase_cost  # add the purchase cost to the total outflow
                inventory += injected_volume  # add the injected gas to the stored volume for future withdrawals
                breakdown.append(
                    {
                        "date": event_date,
                        "type": "injection",
                        "volume": injected_volume,
                        "price": price,
                        "cash_flow": -purchase_cost,
                        "inventory_before": inventory - injected_volume,
                        "inventory_after": inventory,
                    }
                )
            else:
                breakdown.append(
                    {
                        "date": event_date,
                        "type": "injection_skipped",
                        "volume": 0.0,
                        "price": price,
                        "cash_flow": 0.0,
                        "inventory_before": inventory,
                        "inventory_after": inventory,
                    }
                )

        if is_withdrawal:  # if the event is a withdrawal, the holder sells gas and removes it from inventory
            idx = withdrawal_dates.index(event_date)  # find the matching withdrawal date so the correct price can be attached to the event
            price = withdrawal_prices[idx] if withdrawal_prices is not None else estimate_price(event_date)  # use the supplied price if provided, otherwise estimate it with the model
            withdrawn_volume = min(withdrawal_rate, inventory)  # only withdraw as much as is actually stored, preventing negative inventory
            if withdrawn_volume > 0:
                sale_proceeds = withdrawn_volume * (price - withdrawal_cost_per_unit)  # add the sale proceeds and subtract any withdrawal fee from the contract value
                cash_in += sale_proceeds  # add the sell proceeds to the total inflow
                inventory -= withdrawn_volume  # remove the withdrawn gas from storage inventory
                breakdown.append(
                    {
                        "date": event_date,
                        "type": "withdrawal",
                        "volume": withdrawn_volume,
                        "price": price,
                        "cash_flow": sale_proceeds,
                        "inventory_before": inventory + withdrawn_volume,
                        "inventory_after": inventory,
                    }
                )
            else:
                breakdown.append(
                    {
                        "date": event_date,
                        "type": "withdrawal_skipped",
                        "volume": 0.0,
                        "price": price,
                        "cash_flow": 0.0,
                        "inventory_before": inventory,
                        "inventory_after": inventory,
                    }
                )

        previous_date = event_date  # advance the time marker so the next interval is measured against the most recent event date

    value = cash_in - buy_cost - storage_cost  # combine the inflows and outflows into the contract value
    if return_details:
        return {
            "value": float(value),
            "buy_cost": float(buy_cost),
            "cash_in": float(cash_in),
            "storage_cost": float(storage_cost),
            "final_inventory": float(inventory),
            "breakdown": breakdown,
        }
    return float(value)  # return the total contract value after all buy, sell, and storage cash flows have been accounted for


def run_self_tests() -> None:  # define a small built-in self-test so the script checks itself before it is used
    assert abs(estimate_price(date(2020, 10, 31)) - 10.1) < 1e-9  # confirm that the first historical observation is returned unchanged for a known point
    assert abs(estimate_price(date(2024, 9, 30)) - 11.8) < 1e-9  # confirm that the last historical observation is returned unchanged for a known point
    assert estimate_price(date(2025, 3, 31)) > 0  # confirm that a future date produces a positive forecast value
    _, _, sine_coef, cosine_coef, _, _ = _fit_trend_and_seasonal_model()  # confirm that the new sine-cosine seasonal fit can be estimated from the observed history
    assert abs(sine_coef) + abs(cosine_coef) > 0.0  # verify that the seasonal fit actually captures non-zero annual structure

    sample_value = price_storage_contract(
        injection_dates=[date(2024, 1, 31)],
        withdrawal_dates=[date(2024, 4, 30)],
        injection_prices=[10.0],
        withdrawal_prices=[12.0],
        injection_rate=1_000_000.0,
        withdrawal_rate=1_000_000.0,
        max_storage=2_000_000_000.0,
    )
    assert sample_value > 0  # confirm that a straightforward positive-spread storage case produces a positive contract value

    details = price_storage_contract(
        injection_dates=[date(2024, 1, 31)],
        withdrawal_dates=[date(2024, 4, 30)],
        injection_prices=[10.0],
        withdrawal_prices=[12.0],
        injection_rate=1_000_000.0,
        withdrawal_rate=1_000_000.0,
        max_storage=2_000_000_000.0,
        return_details=True,
    )
    assert details["value"] > 0  # confirm that the detailed breakdown still produces a positive contract value
    assert "breakdown" in details  # confirm that the detail view exposes a timeline of the contract cash flows


if __name__ == "__main__":
    import argparse  # import argparse so the script can accept a date from the command line
    import matplotlib.pyplot as plt  # import matplotlib so the script can draw the historical series and the forecasted outlook

    run_self_tests()  # run the built-in checks before the script does any user-facing work
    print("Self-tests passed.")  # print a short message confirming that the internal checks succeeded

    parser = argparse.ArgumentParser(description="Estimate natural gas price from monthly snapshot data")  # create the argument parser with a description of the tool
    parser.add_argument("date", type=str, help="Date to estimate (YYYY-MM-DD)")  # define the required date argument that the user passes to the script
    parser.add_argument("--plot", action="store_true", help="Generate a plot of the series")  # define an optional flag that tells the script to save a chart
    args = parser.parse_args()  # parse the command-line arguments into usable Python objects

    target = datetime.strptime(args.date, "%Y-%m-%d").date()  # convert the text date argument into a Python date object using the expected format
    print(f"Estimated price on {target}: {estimate_price(target):.2f}")  # print the estimated price for the requested date with two decimal places

    if args.plot:  # if the user requested a plot, generate the chart and save it to disk
        import pandas as pd  # import pandas because it makes the small DataFrame for plotting very easy

        series = pd.DataFrame({"Date": DATES, "Price": PRICES})  # build a DataFrame that contains the observed dates and prices for the chart
        plt.figure(figsize=(10, 4))  # create a figure with a wide, readable shape for the time-series plot
        plt.plot(series["Date"], series["Price"], marker="o", linestyle="-", label="Observed monthly snapshot")  # draw the historical monthly observations as a line with markers
        future_dates = [DATES[-1] + timedelta(days=30 * i) for i in range(1, 13)]  # create 12 future dates, one month apart, to visualize the one-year outlook
        future_prices = [estimate_price(d) for d in future_dates]  # estimate a price for each future date in the forecast horizon
        plt.plot(future_dates, future_prices, linestyle="--", marker="x", label="Extrapolated 1-year outlook")  # draw the forecast as a dashed line so it is visually distinct from the historical data
        plt.xlabel("Date")  # label the x-axis with the word Date
        plt.ylabel("Price")  # label the y-axis with the word Price
        plt.title("Natural gas monthly price series")  # give the plot a descriptive title
        plt.legend()  # show the legend so the historical and forecasted paths are easy to distinguish
        plt.tight_layout()  # tighten the layout so the labels and legend fit neatly in the figure
        plt.savefig("natural_gas_prices.png")  # save the plot as a PNG file in the current working directory
        print("Saved plot to natural_gas_prices.png")  # print a message confirming that the chart was written to disk
