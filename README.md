# Big Data Solutions for Forecasting Air Pollution 🌿

A following repository was created to provide insights and showcase solutions within the thesis framework for the University of Debrecen 2024/2025/2 BSc Computer Science.

## Project Overview

In framework of this study we have also developed a UI to showcase the results and accuracy of each model to forecast air quality (specifically PM10 concentration) using an ensemble of:
- **Tree-based models:** GBDT, DART, LightGBM
- **Sequence models:** LSTM, GRU
- **Traditional approach:** Linear Regression
- **Meta-model:** Neural network that learns from base models' predictions to improve final forecasts.

The pipeline includes:
- Data loading and preprocessing
- Data analysis of AQI and historical meteorological data
- Base models training and evaluation
- Meta-feature construction from base model predictions
- Meta-model training and forecasting
- Exporting results and performance metrics for analysis
- API for each of the base models

---

## Project Structure
<pre> \``` src/ │ ├── backend/ # Core backend code (data, models, utilities) │ │ │ ├── app/ # Application utilities │ │ └── model_unpacking.py # Utility for loading saved models │ │ │ ├── data_load/ # Data loading and preprocessing │ │ ├── base_model_data.py # Base preprocessing for models │ │ ├── data_handle.py # Data handler utilities │ │ └── _features.py # Feature definitions │ │ │ ├── models/ # Machine learning models │ │ ├── ensemble_model.py # Ensemble model (meta-learner) │ │ ├── gbdt.py # Gradient Boosting model │ │ ├── dart.py # DART boosting model │ │ ├── lightgbm.py # LightGBM model │ │ ├── lstm_v2.py # LSTM sequence model │ │ └── gru.py # GRU sequence model │ │ │ └── utils/ # (Optional) Utility functions │ └── logger.py # Logging utilities (future improvement) │ ├── ml_model/ # Saved trained models │ ├── lstm_model_v2.keras # Saved LSTM model │ └── ensemble_model.keras # Saved ensemble model │ ├── notebooks/ # Jupyter Notebooks for experiments │ └── ensemble_model_final.ipynb # Final ensemble training notebook │ ├── webapp/ # API layer (Flask app) │ └── src/ # Flask source code │ └── backend/app/ # Flask endpoints for model inference │ ├── requirements.txt # Project dependencies ├── README.md # Project documentation └── .gitignore # Files to ignore in version control
 \``` </pre>



