# User Behavior Analysis

A Python project for analyzing user activity data.  
The project explores user behavior through funnel analysis, retention tracking, and churn prediction features.

## Features

- Data cleaning and preprocessing (ETL)
- Conversion funnel analysis (View → Cart → Purchase)
- User retention analysis
- Churn detection and feature engineering
- Visualization of user behavior metrics

## Project Structure

src/
  etl.py        # data cleaning
  funnel.py     # funnel analysis
  retention.py  # retention analysis
  churn.py      # churn analysis

data/
  events.csv
  events_clean.csv

reports/
  funnel_chart.png
  retention_heatmap.png
  churn_dataset.csv

## Installation

pip install pandas matplotlib

## Run

python src/etl.py
python src/funnel.py
python src/retention.py
python src/churn.py

## Technologies

- Python
- Pandas
- Matplotlib
