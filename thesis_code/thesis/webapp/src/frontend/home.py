import streamlit as st
import pandas as pd
import requests

BASE_URL = 'http://localhost:5001'

WHO_RECOMMEND = {
        "Yearly average": "15μg/m3",
        "Daily average": "45 μg/m3"
    }

def create_map():
    data = {
        "location": ["Debrecen", "Hajnal utca", "Kalotaszeg tér"],
        "lat": [47.5265, 47.5187, 47.5307],
        "lon": [21.6273, 21.6301, 21.6392],
        "color": [[0, 0, 255], [255, 0, 0], [255, 0, 0]]
    }

    df = pd.DataFrame(data)

    # Create the map
    st.map(df,
           latitude='lat',
           longitude='lon',
           color='color',
           size=20,
           zoom=12)

    st.caption("Map showing Debrecen (blue) with two monitoring stations (red)")

def yearly_stats():
    resp = requests.post(BASE_URL + '/api/download_report')
    if resp.status_code != 200:
        st.error("Did not receive response from API :(")
    else:
        df = pd.DataFrame(resp.json()['data'])
        df.set_index('year', inplace=True)
        plot_stats(df)

def plot_stats(df):
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Station1 - Hajnal utca")
        st.bar_chart(df[['Station1_PM10']])
    with cols[1]:
        st.subheader("Station2 - Kalotaszeg tér")
        st.bar_chart(df[['Station2_PM10']])

def main():
    container = st.container(border=True)
    container.markdown(
        """
        Hello there, Data Science enthusiast! Welcome to web app - ***PAP (Predict Air Pollution)***,
        where you can select one of the ML model from available dropdown list and explore the analysis 
        on prediction and download results. See **WHO**'s recommendation on PM10 concentration level below 
        in the table.
        """
    )

    who_recommends = pd.DataFrame([WHO_RECOMMEND])
    container.dataframe(
        who_recommends,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("""## PM10 levels in Debrecen, Hungary""")
    cols = st.columns(2)
    with cols[0]:
        create_map()
    with cols[1]:
        st.markdown("""
            ***Debrecen*** - second largest city in Hungary. In the interactive map on the right
            you can see two monitoring stations from which we collected historical data from 2014 to 2024
            in order to do a research on the forecasting Particulate Matter 10 concentration levels using
            Big Data and Machine Learning Techniques.
            
            :green[***PM10***] - refers to Particulate Matter with a diameter of 10 micrometers. These tiny airborne particles
            come from various ources like:
            - Dust
            - Pollen
            - Vehicle emissions
            - Industrial pollution
            
            PM10 can enter lungs which can cause breathing problems and allergies, asthma etc.
            To prevent the increase in the number of victims due to low air quality, ML and Big data provides
            a solution to predict the peaks of PM10 above WHO's recommendation threshold.  
        """)

    st.markdown("""
    <p style="color:rgba(128, 128, 128, 0.6); font-size:24px; font-family: "Source Sans Pro">
    Yearly average PM10 values for both stations
    </p>
    """, unsafe_allow_html=True)

    yearly_stats()
