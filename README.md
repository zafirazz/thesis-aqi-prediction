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

<img width="209" alt="Screenshot 2025-04-13 at 19 08 27" src="https://github.com/user-attachments/assets/648e31ae-8594-4af7-af23-d802a9fa0e32" />

## How to run:

Follow these steps to set up and run the web application:

1. Clone the repository
```   
git clone https://github.com/yourusername/air-quality-ensemble.git
cd webapp/src/backend
```
3. Create and activate a virtual environment.

For macOS / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
With the virtual environment activated, install the required Python packages:
```
pip install -r requirements.txt
pip install -e .
```

4. Run the backend Flask API
Navigate to the backend application folder and start the Flask server:
```
cd backend
python app.py
```
The Flask API will start locally at:
```
http://127.0.0.1:5000/
```
You can use tools like Postman or cURL to interact with the API endpoints.

5. Run the Streamlit dashboard (frontend)
In a new terminal window (keep the Flask server running), start the Streamlit application in the frontend directory:
```
streamlit run main.py
```
The Streamlit app will open in your browser, typically at:
```
http://localhost:8501/
```
Here, you can interact with the application UI to trigger model training, view predictions, and visualize results.






