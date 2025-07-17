# SGLN Market Trend Fill Modelling Project

This project analyses weekly trends in the SGLN ETF and evaluates the likelihood of an order being filled after a trend signal. It combines basic technical indicators with a decision tree classifier to simulate realistic trading decisions.

---

## Project Overview

The goal is to classify each week as a **Buy**, **Sell**, or **Sideways** trend based on future price movement, then assess whether a hypothetical order placed on that signal would be successfully filled. This approach is useful for testing the realism of trend-based strategies that rely on order execution, not just price direction.

The final output includes:
- Trend classification using EMAs and MACD histogram
- Evaluation of trend model performance
- Analysis of fill rates for buy/sell signals
- Saved plots and classification reports

---

## Project Structure

```text
market_trend_fill_model/
├── prepare_data.py       # Load and clean SGLN price data
├── features.py           # Generate EMAs and MACD histogram
├── model_trend.py        # Label trends and train decision tree classifier
├── model_fill.py         # Simulate order placement and analyse fill success
├── wrap_up.py            # Save plots, export evaluation reports
├── run_project.py        # Entry-point: runs full pipeline
├── outputs/
│   ├── data/             # CSVs, classification reports
│   └── figures/          # Saved visual plots
└── README.md
```


---

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

To execute the full trend and fill modelling pipeline, run:

```bash
python run_project.py
```

All outputs will be saved automatically to the `outputs/` folder.

---

## Status

This repository contains a fully working prototype, modularised into clean Python scripts. Results are saved, but no external datasets are required — all inputs are internal or preprocessed.