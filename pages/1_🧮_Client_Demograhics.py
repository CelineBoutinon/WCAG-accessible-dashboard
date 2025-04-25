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
    st.markdown(
        """
        <style>
        body {
            background-color: black !important;
            color: white !important;
            font-size: 20px !important;
        }
        /* Customize buttons, inputs, etc. for high contrast */
        button, input {
            background-color: yellow !important;
            color: black !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    # Default or light theme CSS or no override
    pass

# For graphs, use color-blind-friendly palettes conditionally
import plotly.express as px

colorblind_palette = px.colors.qualitative.Safe  # colorblind-friendly palette
default_palette = px.colors.qualitative.Plotly

palette = colorblind_palette if accessibility_mode else default_palette



# Get selected client application id
selected_value = st.session_state.get("selected_value", None)
if selected_value is None:
    st.warning("Please select a client credit application reference on the Home page first.")
else:
    st.write(f"Using selected client application: {selected_value}")
    

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

st.image("bandeau.png")


# Send a get request to the API using the selected client credit application reference
# app_response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}") # web API
app_response = requests.get(f"http://127.0.0.1:5000/predict/{selected_value}") # local API

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
    # st.markdown("<h4 style='font-size: 28px;'>Select client demographics to display:</h4>", unsafe_allow_html=True)
    st.write("## ✌️ Step 2 - Select client demographics to display:")
    selected_demographics = st.multiselect("", # Leave text empty to avoid duplicate with above
                                            options=list(client_info.keys()),
                                            default=list(client_info.keys())  # Show all by default
                                            )
    st.write("### You selected client demographics:")
    for demo in selected_demographics:
        st.markdown(f"<span style='font-size:28px;'> - **{demo}:** {client_info[demo]}</span>",
                  unsafe_allow_html=True)
   





