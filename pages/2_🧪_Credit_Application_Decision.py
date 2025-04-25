import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Get selected client application id
selected_value = st.session_state.get("selected_value", None)
if selected_value is None:
    st.warning("Please select a client credit application reference on the Home page first.")
else:
    st.write(f"Using selected client application: {selected_value}")

# Check if API data is loaded
    if "app_data" not in st.session_state:
        st.warning("API data not loaded yet. Please go back to Home page and select a client.")
    else:
        # Access stored data
        shap_values_client_dict = st.session_state.shap_values_client_dict
        shap_values_array = st.session_state.shap_values_array
        feature_names = st.session_state.feature_names
        base_value = st.session_state.base_value
        threshold_value = st.session_state.threshold_value
        client_data_dict = st.session_state.client_data_dict
        client_data_array = st.session_state.client_data_array
        app_data = st.session_state.app_data
        app_response = st.session_state.app_response

        # st.write(f"Using client application {selected_value}")
        # Example: show feature names and SHAP values
        # st.write("Feature names:", feature_names)
        # st.write("SHAP values:", shap_values_array)

        st.write("# 💫 Client credit scoring model results:💫")
        # st.write(f"Client default probability: {app_data['Client default probability'] * 100:.2f}%")
        # st.write("Class :", app_data['Class'])
        if app_data['Class'] == 'no default':
            st.markdown(f"<span style='font-size:28px;'> **Predicted behavior** : client will repay loan 👍</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='font-size:28px;'> **Predicted behavior** : client will default on loan 👎</span>",
                        unsafe_allow_html=True)
        
        # st.write("Decision :", app_data['Decision'])
        if app_data['Decision'] == "grant loan":
            st.write(f"<span style='font-size: 28px;'> **Decision** : {app_data['Decision']} 🥂🎈🎉</span>",
                        unsafe_allow_html=True)
        else:
            st.write(f"<span style='font-size: 28px;'> **Decision** : {app_data['Decision']} ⛔</span>",
                        unsafe_allow_html=True)



        # Display client default probability on a gauge with color change above custom threshold
        bar_color = 'red' if app_data['Client default probability'] > threshold_value else 'forestgreen'
        fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = app_data['Client default probability'] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "forestgreen"}},
        title = {'text': "<b>Client default probability %</b>",  'font': {'size': 28, 'color': 'black'}},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': bar_color},
                'steps' : [{'range': [0, 50], 'color': "palegreen"},
                           {'range': [50, 100], 'color': "lightcoral"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.502, 'value': 50}
                }))

        st.plotly_chart(fig)

        # Create SHAP waterfall plot
        if shap_values_array is not None:
            shap_explanation = shap.Explanation(values=shap_values_array, 
                                                base_values=base_value,
                                                feature_names=feature_names)
            fig, ax = plt.subplots(figsize=(10,6))
            st.title(f"Key decision factors for client {selected_value}")
            shap.plots.waterfall(shap_explanation, max_display=6) # Show the top 5 features and group the remaining features
            st.pyplot(fig)

        else: 
            st.error(f"Failed to fetch Shap values for client application. API status code : {app_response.status_code}")