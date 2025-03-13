# Clustering Ticker Selection Based on Efficiency Ratio Analysis

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Author](#author)

## Introduction

This repository implements clustering techniques to group financial instruments based on their Efficiency Ratio (ER) characteristics. The ER measures the directional movement of price relative to total price movement over a specified period, helping identify instruments with similar trending behaviors.

The main objectives are to:
- Group similar financial instruments based on their efficiency ratios
- Identify optimal trading instruments for different trading strategies
- Reduce the complexity of portfolio selection
- Optimize trading performance through instrument selection

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/jhontd03/ClusteringTickerSelection.git
cd ClusteringTickerSelection
pip install -r requirements.txt
```

### Requirements

The project requires Python 3.11.9 and the following key dependencies:

Key dependencies include:
- MetaTrader5: For market data retrieval
- pandas_ta: For technical analysis calculations
- scikit-learn: For clustering algorithms
- yellowbrick: For cluster visualization
- matplotlib: For data visualization

## Usage

Here's a basic example of how to use the `EfficiencyRatioAnalysis`:

```python
from efr_classification import EfficiencyRatioAnalysis
from datetime import datetime as dt

# Define your tickers
tickers = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

# Configure analysis parameters
start_date = "2015-01-01"
end_date = dt.strftime(dt.now(), "%Y-%m-%d")
time_frame = '1h'

# Create analysis instance
analysis = EfficiencyRatioAnalysis(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    time_frame=time_frame,
    init_period=6,
    end_period=48,
    n_clusters=3
)

# Generate and display clustering results
analysis.graph_clusters_er()
```

## Repository Structure

```
.
│   README.md
│   main.py                  # Main script to run the analysis
│   efr_classification.py    # Core efficiency ratio analysis
│   function_cluster.py      # Clustering algorithms implementation
│   data_loader.py          # Data loading utilities
│   requirements.txt         # Project dependencies
```

## Features

- **Multiple Data Sources**:
  - MetaTrader5 integration
  - Yahoo Finance support
  - CSV file loading capability

- **Efficiency Ratio Analysis**:
  - Multi-period ER calculation
  - Customizable period ranges
  - Mean ER computation

- **Advanced Clustering**:
  - K-means clustering
  - Gaussian Mixture Models (GMM)
  - Agglomerative clustering
  - Automatic optimal cluster selection

- **Visualization**:
  - Cluster visualization
  - ER distribution plots
  - Interactive charts

- **Data Processing**:
  - Automated data cleaning
  - Missing data handling
  - Period normalization
  - Multi-timeframe support

## Author

Jhon Jairo Realpe

jhon.td.03@gmail.com