import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# --- Accessibility toggle (WCAG 1.4.1, 1.4.3 & 1.4.4)---
if "accessibility_mode" not in st.session_state:
    st.session_state.accessibility_mode = False

# --- WCAG: Set unique and descriptive browser tab title (2.4.2) ---
st.set_page_config(page_title="Credit Scoring App - Home",
                   page_icon="🏠",
                   layout="wide")

# st.markdown("# Accessibility Toggle")  # Large label above the toggle
st.session_state.accessibility_mode = st.toggle(
    label="Accessibility Toggle - click here to enable high-contrast, larger fonts & colorblind-friendly color scheme",
    value=st.session_state.accessibility_mode,
    help="Toggle to switch to high contrast and larger fonts for better accessibility (High Contrast, Larger Fonts," \
    " Color-blind Friendly)",
    key="accessibility_toggle")

if not st.session_state.accessibility_mode:
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Arial; font-size:3em;'> " \
    "🔍 Click toggle on top left of page" \
    " to enable Accessibility Mode 🔍 </h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Arial; font-size:1em;'> " \
    "🔍 Click toggle on top left of page" \
    " to disable Accessibility Mode 🔍 </h1>", unsafe_allow_html=True) 

def set_default_theme():
    st.markdown("""<style> body {font-size: 1em; color: #000000; background-color: white; }
                </style>""", unsafe_allow_html=True)

def set_high_contrast_theme():
    st.markdown("""<style> body {font-size: 1.25em !important; color: #FFFFFF !important; 
                background-color: #000000 !important;}.stText, .stMarkdown {font-size: 1.25em !important;}
                button {background-color: yellow !important; color: black !important;font-weight: bold;}
                </style>""", unsafe_allow_html=True)

if st.session_state.accessibility_mode:
    set_high_contrast_theme()
else:
    set_default_theme()

# --- Set default background color (WCAG 1.4.1) ---
def set_bg_color(color):
    st.markdown(f"""<style>.stApp {{background-color: {color};}} </style>""", unsafe_allow_html=True)

set_bg_color('#fbf0ef')  # light pink

# --- Display title & company logo with alt text (WCAG 1.1.1) ---
if st.session_state.accessibility_mode:
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Walbaum Heading; font-size:4em;'> Welcome to the </h1>",
                unsafe_allow_html=True)
    st.image("logo_bw.png", caption="Company Logo: Prêt à Dépenser", use_container_width=True)
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Walbaum Heading; font-size:4em;'> Credit Scoring App! </h1>",
                unsafe_allow_html=True)
    st.write("# You can toggle between the Home page and the 4 pages of this app with TAB + ENTER")
    st.write("")
    st.write("")

else:
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Walbaum Heading; font-size:2em;'> Welcome to the </h1>",
                unsafe_allow_html=True)
    st.image("logo_streamlit.png", use_container_width=False)  # Added alt text via caption
    st.markdown("<h1 style='color:#242164; text-align:center; font-family:Walbaum Heading; font-size:2em;'> Credit Scoring App! </h1>",
                unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")


# --- Step 1: Get user to select client credit application reference ---
if st.session_state.accessibility_mode:
    st.write("# ☝🏿 Step 1 - Enter a client credit application reference:")
    if "selected_value" not in st.session_state:
        st.session_state.selected_value = 1  # default value
    st.session_state.selected_value = st.number_input(
        label="",
        min_value=1,
        max_value=48745,
        value=st.session_state.selected_value,
        key="selected_value_input"
    )
    selected_value = st.session_state.selected_value
    st.write(f"# You selected client application: {selected_value}")
    st.write("")
    st.write("")

else:
    st.write("### ☝️ Step 1 - Enter a client credit application reference:")
    if "selected_value" not in st.session_state:
        st.session_state.selected_value = 1  # default value
    st.session_state.selected_value = st.number_input(
        label="",
        min_value=1,
        max_value=48745,
        value=st.session_state.selected_value,
        key="selected_value_input"
    )
    selected_value = st.session_state.selected_value
    st.write(f"### You selected client application: {selected_value}")
    st.write("")
    st.write("")


# --- Fetch API data only if not already fetched for this selected_value ---
if ("last_fetched_value" not in st.session_state) or (st.session_state.last_fetched_value != selected_value):
    app_response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}")
    app_data = app_response.json()

    if app_response.status_code == 200:
        shap_values_client_json = app_data["Shap values client"]
        shap_values_client_dict = json.loads(shap_values_client_json)[0]
        shap_values_array = np.array(list(shap_values_client_dict.values()))
        feature_names = list(shap_values_client_dict.keys())
        base_value = app_data.get("Expected Shap Value")
        threshold_value = app_data.get("Threshold")
        client_data_json = app_data["Client data"]
        client_data_dict = json.loads(client_data_json)[0]
        client_data_array = np.array(list(client_data_dict.values()))

        st.session_state.app_response = app_response
        st.session_state.app_data = app_data
        st.session_state.shap_values_client_dict = shap_values_client_dict
        st.session_state.shap_values_array = shap_values_array
        st.session_state.feature_names = feature_names
        st.session_state.base_value = base_value
        st.session_state.threshold_value = threshold_value
        st.session_state.client_data_dict = client_data_dict
        st.session_state.client_data_array = client_data_array

        st.session_state.last_fetched_value = selected_value
    else:
        st.error(f"Failed to fetch data. API status code : {app_response.status_code}")

else:
    app_data = st.session_state.app_data
    shap_values_client_dict = st.session_state.shap_values_client_dict
    shap_values_array = st.session_state.shap_values_array
    feature_names = st.session_state.feature_names
    base_value = st.session_state.base_value
    threshold_value = st.session_state.threshold_value
    client_data_dict = st.session_state.client_data_dict
    client_data_array = st.session_state.client_data_array

# --- Main function placeholder ---
def main():
    pass

if __name__ == "__main__":
    main()
