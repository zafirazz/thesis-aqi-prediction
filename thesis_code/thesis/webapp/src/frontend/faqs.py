import streamlit as st

def main():
    st.markdown("""### 1. What is PM10?""")
    container = st.container(border=True)
    container.write("""
    PM10 refers to particulate matter (PM) with a diameter of 10 micrometers or less. These are tiny particles 
    suspended in the air that can include dust, dirt, soot, smoke, and liquid droplets. PM10 is a significant air pollutant that can penetrate the lungs and cause various health issues, 
    including respiratory problems, cardiovascular diseases, and reduced lung function. 

    Sources of PM10 include vehicle emissions, industrial activities, construction, burning of fossil fuels, natural sources like wildfires, and dust storms. Many countries monitor PM10 levels as part 
    of their air quality assessment programs, with regulatory standards established to protect public health.
    """)

    st.markdown('### 2. How to know which model is LSTM?')
    container2 = st.container(border=True)
    url_statsquest = 'https://www.youtube.com/watch?v=YCzL96nL7j0&ab_channel=StatQuestwithJoshStarmer'

    video_id = url_statsquest.split("v=")[1].split("&")[0]
    st.write('''I enjoy StatsQuest tutorials about Machine Learning! There are a lot of video tutorials'
             on his YouTube channel and one of them is LSTM. If you want to know more about other models we offer
             I recommend on checking this video below!''')
    st.markdown(f"""
    <iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}?autoplay=0&controls=1" 
    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen></iframe>
    """, unsafe_allow_html=True)