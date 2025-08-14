# Mortgage Analytics Toolkit

A Python-based toolkit for analyzing mortgage-related products, interest rate dynamics, and fixed-income valuation.  
This project combines market data analysis, mathematical finance models, and simulation techniques to support research, trading, and risk management workflows. The toolkit is based on an introduction to mortgage analytics project reached here: 
https://colab.research.google.com/drive/1kBUtBgGQ7uytfb6BrAUgF-zJbG_5mC1F?usp=sharing

## 📌 Features

- **Interactive Demo** — Runs example calculations based on current interest rates from public data sources.
- **Mortgage & Coupon Calculations** — Computes cash flows, yields, durations, and coupon equivalencies for mortgage-backed securities (MBS).
- **Forward Curve Bootstrapping** — Constructs zero-coupon and forward rate curves from market instruments.
- **Principal Component Analysis (PCA)** — Decomposes interest rate curve movements into key factors (e.g., level, slope, curvature).
- **Hull-White Model Simulation** — Generates stochastic interest rate paths for scenario analysis and valuation.
- **Unit Testing Suite** — Ensures code reliability and model accuracy via `pytest` and sample datasets.

---

## 🏗️ Installation

Clone the repository:

```bash
git clone https://github.com/sagan-hartley/mbs_project
cd mbs_project
