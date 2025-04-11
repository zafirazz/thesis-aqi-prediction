from io import BytesIO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

import pandas as pd
import requests
import streamlit as st
from streamlit_option_menu import option_menu

from thesis_code.thesis.webapp.src.frontend import home, predict_analyse, faqs


class Application:
    def __init__(self):
        self.BASE_URL = 'http://localhost:5001'
        self.session_state = st.session_state

    def main(self):
        st.set_page_config(layout="wide", page_title="PAP UI", page_icon="🌱")
        st.markdown("# :green[PM10] Big Data and ML solutions")

        with st.container():
            selected=option_menu(
                menu_title=None,
                options=["Home", "Predict and Analyze", "FAQs"],
                icons=["house", "robot", "question"],
                default_index=0,
                orientation="horizontal",
                styles={
                    "nav-link": {"--hover-color": "87BAAB"},
                    "nav-link-selected": {"background-color": "green"}
                }
            )
            if selected == "Home":
                home.main()
            elif selected == "Predict and Analyze":
                predict_analyse.main()
            else:
                faqs.main()

if __name__ == '__main__':
    app = Application()
    app.main()

