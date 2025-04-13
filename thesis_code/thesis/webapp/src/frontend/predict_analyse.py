from io import BytesIO

import streamlit as st
import requests
import pandas as pd

BASE_URL = 'http://localhost:5001'

def forecast_linreg(model):
    payload = {"model_name": model}
    if model == "Linear Regression":
        with st.spinner("Training model and preparing analysis..."):
            resp = requests.post(BASE_URL + '/api/model/linear_regression')
            st.session_state["prediction"] = resp.json()["data"]
            if resp.status_code == 200:
                st.success("Results are ready!")
            else:
                st.error("Response is empty from api :(")
                return
    plot_pred()

def forecast_lstm(model_name):
    with st.spinner("Training model and preparing analysis..."):
        model_api_map = {
            "LSTM (4 layers)": "/api/model/lstm",
            "LSTM (2 layers)": "/api/model/lstm_reduced",
            "GRU": "/api/model/gru",
            "GBDT": "/api/model/gbdt",
            "DART": "/api/model/dart",
            "LightGBM": "/api/model/lgbm",
            "Ensemble model": "/api/model/ensemble",
        }

        api_endpoint = model_api_map.get(model_name)

        resp = requests.post(BASE_URL + api_endpoint)
        st.session_state["prediction"] = resp.json()["data"]
        if resp.status_code == 200:
            st.success("Results are ready!")
        else:
            st.error("Response is empty from api :(")
            return
    plot_pred()

def plot_pred():
    y_pred = st.session_state["prediction"]["pred"][:200]
    y_test = st.session_state["prediction"]["test"][:200]
    mae = st.session_state["prediction"]["mae"]
    mse = st.session_state["prediction"]["mse"]
    r2 = st.session_state["prediction"]["r2"]

    df_plot = {"PM10_prediction": y_pred, "PM10_actual": y_test,}
    df_errors = {"MAE": mae, "MSE": mse, "R2": r2}

    cols = st.columns(2)

    df = pd.DataFrame(df_plot)

    df = df.reset_index().rename(columns={'index': 'Test Sample Index'})

    with cols[0]:
        st.line_chart(
            df,
            x='Test Sample Index',
            y=['PM10_prediction', 'PM10_actual']
        )

    with cols[1]:
        st.table(df_errors)
        excel_data = download_report(df_plot, df_errors)
        st.download_button(label="Generate report 📥", data=excel_data, file_name="prediction_pm10.xlsx")

def download_report(df_plot, df_errors):
    df_plot = pd.DataFrame(df_plot)
    df_errors = pd.DataFrame([df_errors])
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_plot.to_excel(writer, sheet_name="prediction_of_pm10", index=False)
        df_errors.to_excel(writer, sheet_name="accuracy_of_prediction", index=False)
    output.seek(0)
    return output

def main():
    st.session_state["model_name"] = st.selectbox(
        "Please select a model",
        ("Linear Regression", "LSTM (4 layers)", "LSTM (2 layers)", "GBDT",
         "DART", "Ensemble model", "GRU", "LightGBM"),
    )

    if st.session_state["model_name"] == "Linear Regression":
        if st.button("Predict PM10"):
            forecast_linreg(st.session_state["model_name"])
    elif st.session_state["model_name"] != "":
        if st.button("Predict PM10"):
            forecast_lstm(st.session_state["model_name"])
