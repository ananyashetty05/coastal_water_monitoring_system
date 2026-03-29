# Coastal Water Quality Monitoring System / CoastalWatch

CoastalWatch

CoastalWatch is an interactive Streamlit dashboard designed for monitoring, analyzing, and forecasting coastal water quality across US, Ireland, England and China mainly. It combines environmental data processing, geospatial visualization, machine learning, and time-series forecasting into a unified interface.

⸻

Overview

CoastalWatch enables users to:
	•	Ingest and validate environmental monitoring datasets
	•	Compute and classify CCME Water Quality Index (WQI)
	•	Visualize monitoring stations on an interactive map
	•	Analyze temporal trends and water quality metrics
	•	Forecast water quality parameters for the next 7 days
	•	Generate actionable recommendations for degraded conditions

⸻

Key Features

Data Ingestion & Validation
	•	CSV upload with strict schema validation
	•	Automatic cleaning and type normalization
	•	Missing value handling and consistency checks

Water Quality Classification
	•	CCME WQI computation pipeline
	•	Rule-based fallback classification
	•	Multi-model ML benchmarking:
	•	Logistic Regression
	•	K-Nearest Neighbors (KNN)
	•	Random Forest
	•	Gradient Boosting
	•	Support Vector Machine (SVM)
	•	Voting Classifier
	•	Stacking Classifier

Geospatial Visualization
	•	Interactive map using pydeck
	•	Color-coded markers based on WQI category
	•	Station-level insights and summaries
	•	Filtering by country and waterbody type

Analytics Dashboard
	•	Time-series trend visualization
	•	Parameter-wise analysis (DO, BOD, Nitrate, etc.)
	•	Country-level comparisons
	•	CCME score distribution and rankings

Forecasting Engine
	•	7-day forecasting using OLS regression
	•	Metric-wise prediction
	•	Predicted WQI classification
	•	Future quality insights

Recommendations Engine
	•	Automated suggestions for:
	•	Poor or marginal water quality
	•	Parameter-specific anomalies
	•	Rule-based environmental mitigation guidance

UI & Experience
	•	Responsive Streamlit layout
	•	Modular components for reusability
	•	Custom CSS styling for a clean dashboard feel

________________________________________________________________________________________________________________________________________________


########## *folder structure* ############

coastalwatch/
│
├── app.py                          # entry point, just navigation
│
├── pages/
│   ├── 01_upload.py                # upload & sample data
│   ├── 02_map.py                   # location map view
│   ├── 03_analytics.py             # dashboard & charts
│   └── 04_predictions.py          # 7-day forecast
│
├── core/
│   ├── __init__.py
│   ├── state.py                    # session state helpers
│   ├── classifier.py               # water quality scoring
│   ├── processor.py                # data loading & stats
│   └── predictor.py                # linear regression forecast
│
├── components/
│   ├── __init__.py
│   ├── metric_row.py               # the 5 stat cards row
│   ├── quality_badge.py            # Safe / Moderate / Poor badge
│   └── summary_table.py           # min/max/avg table
│
├── data/
│   └── sample_data.csv             # pre-generated sample (optional)
│
├── assets/
│   └── style.css                   # custom CSS overrides
│
├── requirements.txt
└── README.md

________________________________________________________________________________________________________________________________________________

#### Usage Guide

Step 1: Upload Dataset
	•	Navigate to Upload Data
	•	Upload your CSV file
	•	View dataset preview and validation results

Step 2: Explore Map
	•	Open Map View
	•	Inspect stations geographically
	•	Apply filters by country or waterbody type

Step 3: Analyze Trends
	•	Visit Analytics
	•	Explore parameter trends and WQI distribution

Step 4: Forecast Future Quality
	•	Go to Predictions
	•	View 7-day forecasts for all parameters
	•	Analyze predicted WQI categories

  ______________________________________________________________________________________________________________________________________________

  Future Enhancements
	•	Advanced forecasting models:
	•	ARIMA
	•	Facebook Prophet
	•	LSTM (Deep Learning)
	•	Database integration (PostgreSQL / MongoDB)
	•	User authentication & saved sessions
	•	Real-time data ingestion (APIs / IoT sensors)
	•	Improved geospatial clustering & Mapbox styling

 _______________________________________________________________________________________________________________________________________________

 Tech Stack
	•	Frontend/UI: Streamlit
	•	Data Processing: Pandas, NumPy
	•	Visualization: PyDeck, Matplotlib
	•	Machine Learning: Scikit-learn
	•	Forecasting: Statsmodels (OLS)

  THANK YOU!

  ~ Project by: 1) Ananya Shetty
                2) Soumil Jha
                3) Rochana Deshpande
                4) Shikha Bhanushali
