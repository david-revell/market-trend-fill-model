# Market Trend Fill Modelling with EMA and MACD Features

This project analyses weekly trends in the SGLN ETF and simulates realistic order execution using technical indicators and classification models. It combines EMA and MACD features with decision tree classifiers to support cautious, rules-based ETF trading under real-world constraints.

---

## Project Overview

The goal is to classify each week as a **Buy**, **Sell**, or **Sideways** trend based on future price movement, and assess whether a directional limit order placed on that signal would have been filled. This two-stage system helps test the practicality of trend-based strategies where fill success is as important as trend accuracy.

**Key components:**
- Trend classification using lagged EMAs and MACD histogram
- Decision tree models for trend prediction and fill likelihood
- Visual summaries and classification reports for both stages

*Full PDF report included in the repo.*

---

## Project Structure

``` text
market_trend_fill_model/
├── prepare_data.py       # Load and clean SGLN price data
├── features.py           # Generate EMAs and MACD histogram
├── model_trend.py        # Label trends and train classifier
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

``` bash
pip install -r requirements.txt
``` 

---

## Running the Project

To execute the full trend and fill modelling pipeline:

``` bash
python run_project.py
``` 

Outputs (data and figures) are saved automatically to the `outputs/` directory.

---

## Status

This is a fully working prototype, structured into modular Python scripts. No external data is needed — all inputs are self-contained or generated from processed SGLN price data.

