# !!! THIS IS THE OCDS P8 APP !!!

# To run app locally, navigate to : C:\Users\celin\DS Projets Python\OCDS-repos-all\OCDS-P8-API
# in the command line and run : py -m streamlit run streamlit_cloud_app_P8_v1.py
# App runs on Streamlit Community Cloud at <TBD>

# IMPORTANT: In advanced settings, choose Python 3.10 when deploying the app in Streamlit Community Cloud
# to avoid errors related to distutils (discontinued from Python 3.12 onwards).

import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# WCAG toggle
if "accessibility_mode" not in st.session_state:
    st.session_state.accessibility_mode = False

# Toggle widget to activate accessibility mode
# st.write("# 🔍 Enable Accessibility Mode 🔍")
st.markdown("<h1 style='color:#242164; text-align:center; font-family:Arial; font-size:3em;'> 🔍 Enable Accessibility Mode 🔍 </h1>",
            unsafe_allow_html=True)
st.session_state.accessibility_mode = st.toggle(
    label="     ",
    value=st.session_state.accessibility_mode,
    help="# Toggle to switch to high contrast and larger fonts for better accessibility (High Contrast, Larger Fonts, Color-blind Friendly)",
    key="accessibility_toggle"
)

if st.session_state.accessibility_mode:
    # st.write("# Accessibility mode is ON")
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Arial; font-size:3em;'> Accessibility mode is ON </h1>",
            unsafe_allow_html=True)
else:
    # st.write("# Accessibility mode is OFF")
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Arial; font-size:3em;'> Accessibility mode is OFF </h1>",
            unsafe_allow_html=True)

def set_default_theme():
    st.markdown(
        """
        <style>
        body {
            font-size: 16px;
            color: #000000;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_high_contrast_theme():
    st.markdown(
        """
        <style>
        body {
            font-size: 20px !important;
            color: #FFFFFF !important;
            background-color: #000000 !important;
        }
        /* Example: make all text larger */
        .stText, .stMarkdown {
            font-size: 20px !important;
        }
        /* Adjust button colors */
        button {
            background-color: yellow !important;
            color: black !important;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.accessibility_mode:
    set_high_contrast_theme()
else:
    set_default_theme()

# Set default background color
def set_bg_color(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage: set the background color to lightblue
set_bg_color('#fbf0ef') # light pink
# set_bg_color('#2fbeb5') # light green
# set_bg_color('#f1bd5f') # sand

# st.set_page_config(
#     page_title="Hello",
#     page_icon="🏠",
# )


# Display title & company logo
st.markdown("<h1 style='color:#242164; text-align:center; font-family:Walbaum Heading; font-size:3em;'> Welcome to the </h1>",
            unsafe_allow_html=True)
st.image("logo_streamlit.png")
st.markdown("<h1 style='color:#242164; text-align:center; font-family:Walbaum Heading; font-size:3em;'> Credit Scoring App! </h1>",
            unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")

# Get user to select client credit application reference
st.write("## ☝️ Step 1 - Enter a client credit application reference:")
# selected_value = st.number_input("", min_value=1, max_value=48745)
if "selected_value" not in st.session_state:
    st.session_state.selected_value = 1  # default value

st.session_state.selected_value = st.number_input(
    "Enter client credit application reference:",
    min_value=1,
    max_value=48745,
    value=st.session_state.selected_value,
    key="selected_value_input"
)

selected_value = st.session_state.selected_value


# Display the selected client credit application reference
st.write(f"### You selected client application: {selected_value}")
st.write("")
st.write("")


# --- Fetch API data only if not already fetched for this selected_value ---
if ("last_fetched_value" not in st.session_state) or (st.session_state.last_fetched_value != selected_value):
    # Make API call
    # app_response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}") # web API
    app_response = requests.get(f"http://127.0.0.1:5000/predict/{selected_value}") # local API
    app_data = app_response.json()

    if app_response.status_code == 200:
        # Parse API data
        shap_values_client_json = app_data["Shap values client"]
        shap_values_client_dict = json.loads(shap_values_client_json)[0]
        shap_values_array = np.array(list(shap_values_client_dict.values()))
        feature_names = list(shap_values_client_dict.keys())
        base_value = app_data.get("Expected Shap Value")
        threshold_value = app_data.get("Threshold")
        client_data_json = app_data["Client data"]
        client_data_dict = json.loads(client_data_json)[0]
        client_data_array = np.array(list(client_data_dict.values()))

        # Store all in session_state
        st.session_state.app_response = app_response
        st.session_state.app_data = app_data
        st.session_state.shap_values_client_dict = shap_values_client_dict
        st.session_state.shap_values_array = shap_values_array
        st.session_state.feature_names = feature_names
        st.session_state.base_value = base_value
        st.session_state.threshold_value = threshold_value
        st.session_state.client_data_dict = client_data_dict
        st.session_state.client_data_array = client_data_array

    # Track which value was last fetched to avoid redundant calls
        st.session_state.last_fetched_value = selected_value

    else:
        st.error(f"Failed to fetch data. API status code : {app_response.status_code}")

else:
    # Use cached data from session_state
    app_data = st.session_state.app_data
    shap_values_client_dict = st.session_state.shap_values_client_dict
    shap_values_array = st.session_state.shap_values_array
    feature_names = st.session_state.feature_names
    base_value = st.session_state.base_value
    threshold_value = st.session_state.threshold_value
    client_data_dict = st.session_state.client_data_dict
    client_data_array = st.session_state.client_data_array
  
# Main function placeholder
def main():
    pass

if __name__ == "__main__":
    main()