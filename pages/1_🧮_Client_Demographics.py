import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


accessibility_mode = st.session_state.get("accessibility_mode", False)

if accessibility_mode:
    # Inject high contrast CSS, increase font sizes, etc.
    st.markdown("""<style> body {background-color: black !important; color: white !important;
                font-size: 20px !important;} 
                button, input {background-color: yellow !important; color: black !important;
                font-weight: bold !important;} </style> """, unsafe_allow_html=True)
else:
    pass

# Get selected client application id
selected_value = st.session_state.get("selected_value", None)

HOME_PAGE = "streamlit_cloud_app_P8_v1.py"

if st.session_state.accessibility_mode:
    if selected_value is None:
        st.warning("# Please select a client credit application reference on the Home page first.")
        st.markdown("# 🏠︎ [Back to Home Page](streamlit_cloud_app_P8_v1.py)")
        st.page_link(HOME_PAGE, label="Back to Home Page")
    else:
        st.write(f"# Using selected client application: {selected_value}")
        st.write("# 🗣 Accessibility mode enabled - navigate to home page to disable")
        st.markdown("# 🏠︎ [Back to Home Page](streamlit_cloud_app_P8_v1.py)")
        st.page_link(HOME_PAGE, label="Back to Home Page")
else:
    if selected_value is None:
        st.warning("### Please select a client credit application reference on the Home page first.")
        st.page_link(HOME_PAGE, label="🏠 Back to Home Page")
    else:
        st.write(f"### Using selected client application: {selected_value}")
        st.write("### 🛈 Accessibility mode disabled - navigate to home page to enable")
        st.page_link(HOME_PAGE, label="🏠 Back to Home Page")

# --- Set default background color (WCAG 1.4.1) ---
def set_bg_color(color):
    st.markdown(f"""<style>.stApp {{background-color: {color};}} </style>""", unsafe_allow_html=True)

set_bg_color('#fbf0ef')  # light pink

if st.session_state.accessibility_mode:
    st.image("bandeau_bw.png", caption="Company Logo: Prêt à Dépenser", use_container_width=True)  # Added alt text via caption
else:
    st.image("bandeau.png", use_container_width=False)

# Send a get request to the API using the selected client credit application reference
app_response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}") # web API
# app_response = requests.get(f"http://127.0.0.1:5000/predict/{selected_value}") # local API

# Import elements from API response separately for graphs
app_data = app_response.json()
shap_values_client_json = app_data["Shap values client"]
shap_values_client_dict = json.loads(shap_values_client_json)[0]
shap_values_array = np.array(list(shap_values_client_dict.values()))
feature_names = list(shap_values_client_dict.keys())
base_value = app_data.get("Expected Shap Value")
threshold_value =  app_data.get("Threshold")
client_data_json = app_data["Client data"]
client_data_dict = json.loads(client_data_json)[0]
client_data_array = np.array(list(client_data_dict.values()))

# Display user's chosen basic demographics from API response
if app_response.status_code == 200:
    client_info = app_data['Client summary information'][0]

    if st.session_state.accessibility_mode:
        st.write("# ✌🏿 Step 2 - Select client demographics to display:")
        selected_demographics = st.multiselect("", options=list(client_info.keys()),
                                               default=list(client_info.keys())) # Show all by default
                                                 # !!! Streamlit's st.multiselect widget does not currently support 
                                                  # direct customization of the color or style of the options or the 
                                                  # selected tags within the dropdown, however thios shade of red with
                                                  # white writing is considered accessible to most types of color-vision
                                                  # deficiencies. Cf. note on SHAP waterfall plot 13/07/25.
        for demo in selected_demographics:
            st.markdown(f"<span style='font-size:40px; color:#000000;'> - **{demo}:** {client_info[demo]}</span>",
                    unsafe_allow_html=True)
        
    else: 
    # st.markdown("<h4 style='font-size: 28px;'>Select client demographics to display:</h4>", unsafe_allow_html=True)
        st.write("### ✌️ Step 2 - Select client demographics to display:")
        selected_demographics = st.multiselect("", options=list(client_info.keys()),
                                               default=list(client_info.keys())) # show all by default
        st.write("### You selected the following client demographics:")
        for demo in selected_demographics:
            st.markdown(f"<span style='font-size:28px;'> - **{demo}:** {client_info[demo]}</span>",
                    unsafe_allow_html=True)
   





