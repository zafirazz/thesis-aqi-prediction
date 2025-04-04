from io import BytesIO

import pandas as pd
import requests
import streamlit as st

WHO_RECOMMEND = {
        "Yearly average": "15μg/m3",
        "Daily average": "45 μg/m3"
    }

class Application:
    def __init__(self):
        self.BASE_URL = 'http://localhost:5001'
        self.session_state = st.session_state

    def forecast_linreg(self, model_name):
        payload = {"model_name": model_name}
        if model_name == "Linear Regression":
            with st.spinner("Training model and preparing analysis..."):
                resp = requests.post(self.BASE_URL + '/api/model/linear_regression')
                self.session_state["prediction"] = resp.json()["data"]
                if resp.status_code == 200:
                    st.success("Results are ready!")
                else:
                    st.error("Response is empty from api :(")
                    return
        self.plot_pred()

    def plot_pred(self):
        y_pred = self.session_state["prediction"]["result"][:365]
        mae = self.session_state["prediction"]["mae"]
        mse = self.session_state["prediction"]["mse"]
        r2 = self.session_state["prediction"]["r2"]

        df_plot = {"PM10_prediction": y_pred}
        df_errors = {"MAE": mae, "MSE": mse, "R2": r2}

        cols = st.columns(2)
        with cols[0]:
            st.line_chart(df_plot)
        with cols[1]:
            st.table(df_errors)
            excel_data = self.download_report(df_plot, df_errors)
            st.download_button(label="Generate report 📥", data=excel_data, file_name="prediction_pm10.xlsx")

    def download_report(self, df_plot, df_errors):
        df_plot = pd.DataFrame(df_plot)
        df_errors = pd.DataFrame([df_errors])
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            df_plot.to_excel(writer, sheet_name="prediction_of_pm10", index=False)
            df_errors.to_excel(writer, sheet_name="accuracy_of_prediction", index=False)
        out.seek(0)
        return out

    def main(self):

        st.set_page_config(layout="wide", page_title="PAP UI", page_icon="🌱")
        st.markdown("# :green[PM10] Forecasting using Big Data and ML")

        container = st.container(border=True)
        container.markdown(
            """
            Hello there, Data Science enthusiast! Welcome to web app - ***PAP (Predict Air Pollution)***,
            where you can select one of the ML model from available dropdown list and explore the analysis 
            on prediction and download results. See WHO's recommendation on PM10 level below in the table.
            """
        )

        who_recommends = pd.DataFrame([WHO_RECOMMEND])
        container.dataframe(
            who_recommends,
            hide_index=True,
            use_container_width=True
        )

        self.session_state["model_name"] = st.selectbox(
            "Please select a model",
            ("Linear Regression", "LSTM (4 layers)", "LSTM (2 layers)", "GBDT",
             "DART", "Ensemble model", "GRU", "LightGBM"),
        )

        if self.session_state["model_name"] != "":
            if st.button("Predict PM10"):
                self.forecast_linreg(self.session_state["model_name"])


if __name__ == '__main__':
    app = Application()
    app.main()

