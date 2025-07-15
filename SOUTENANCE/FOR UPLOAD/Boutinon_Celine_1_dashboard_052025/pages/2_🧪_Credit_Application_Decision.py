import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from PIL import Image
import io

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
        st.write("# 🗣 Accessibility mode disabled - navigate to home page to enable")
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

if not st.session_state.accessibility_mode:
    st.write("## 💫 Client credit scoring model results:💫")
    if app_data['Class'] == 'no default':
        st.markdown(f"<span style='font-size:28px;'> **Predicted behavior** : client will repay loan 👍</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='font-size:28px;'> **Predicted behavior** : client will default on loan 👎</span>",
                    unsafe_allow_html=True)
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
        st.title(f" Key decision factors for client {selected_value}")
        shap.plots.waterfall(shap_explanation, max_display=6) # Show the top 5 features and group the remaining features
        st.pyplot(fig)
    else: 
        st.error(f"Failed to fetch Shap values for client application. API status code : {app_response.status_code}")

    # Display global feature importance
    st.markdown("#### Top 5 decision factors across all clients:")
    st.image("global_feature_importance.png", caption="Top 5 credit application decision factors",
             use_container_width=True)

    st.markdown("""
    ### *Right hand side graph interpretation*

    Across all client applications, and in decreasing order, the top 5 factors influencing the success of a client's credit application are:

    - the current credit rating,
    - the percentage of times a client has not met their monthly minimum repayment amount,
    - the current number of existing credits the client has,
    - the percentage of the proposed down payment for the current credit application; and
    - the gender of the client.
    """)

    st.markdown("""
    ### *Left hand side graph interpretation*

    Colors indicate whether the variable is positively or negatively affecting the outcome of a client's credit application as its value increases. Blue color towards the right hand side of the graph indicates a positive influence. The higher the credit rating and the down payment, the higher the likelihood of a client's application being accepted; a history of underpayments, a relatively large number of other existing credits, or the male gender of the applicant all negatively affect the outcome of the application (red towards the right hand side of the graph).
    """)


else:
    st.write("# ᯓ★ Client credit scoring model results: ★ᯓ")
    if app_data['Class'] == 'no default':
        st.markdown(f"<span style='font-size:40px;'> **Predicted behavior** : client will repay loan ✔ </span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='font-size:40px;'> **Predicted behavior** : client will default on loan ✘ </span>",
                    unsafe_allow_html=True)
        
    if app_data['Decision'] == "grant loan":
        st.write(f"<span style='font-size: 40px;'> **Decision** : {app_data['Decision']} ✔ </span>",
                    unsafe_allow_html=True)
    else:
        st.write(f"<span style='font-size: 40px;'> **Decision** : {app_data['Decision']} ✘ </span>",
                    unsafe_allow_html=True)

    # Display client default probability on a gauge with color change above custom threshold
    bar_color = 'black' if app_data['Client default probability'] > threshold_value else 'dimgray'
    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = app_data['Client default probability'] * 100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    delta = {'reference': 50, 'increasing': {'color': "black"}, 'decreasing': {'color': "dimgray"}},
    title = {'text': "<b>Client default probability %</b>",  'font': {'size': 40, 'color': 'black'}},
    gauge = {'axis': {'range': [None, 100]},
            'bar': {'color': bar_color},
         'steps' : [{'range': [0, 50], 'color': "white"},
                         {'range': [50, 100], 'color': "gainsboro"}],
            'threshold' : {'line': {'color': "black", 'width': 6}, 'thickness': 0.502, 'value': 50}
            }))
    st.plotly_chart(fig)

    # Create SHAP waterfall plot
    shap.plots.colors.red = "dimgray"
    shap.plots.colors.blue = "gainsboro"
    if shap_values_array is not None:
        shap_explanation = shap.Explanation(values=shap_values_array, 
                                            base_values=base_value,
                                            feature_names=feature_names)
        st.title(f"Key decision factors for client {selected_value}")
        shap.plots.waterfall(shap_explanation, max_display=6) # Show the top 5 features and group the remaining features
        fig = plt.gcf()
        st.pyplot(fig)
        # NB: due to limited customisation available with SHAP plots, it was not possible to draw a B&W or greyscale
        # waterfall plot, however the default red & blue colors in SHAP plots are considered colorblind friendly for most
        # types of color vision deficiencies. Numbers are also displayed in white font on the bars, making them clearly
        # readable for colorblind users (black on red background must be avoided though). Test of graph with 1 colorblind
        # user on 13/07/25 - all ok.

    else:
        st.error(f"Failed to fetch Shap values for client application. API status code : {app_response.status_code}")

    # Display global feature importance
    st.markdown("# Top 5 decision factors across all clients:")
    st.image("global_feature_importance.png", caption="Top 5 credit application decision factors",
             use_container_width=True)

    st.markdown("""
    # *Right hand side graph interpretation*

    Across all client applications, and in decreasing order, the top 5 factors influencing the success of a client's credit application are:

    - the current credit rating,
    - the percentage of times a client has not met their monthly minimum repayment amount,
    - the current number of existing credits the client has,
    - the percentage of the proposed down payment for the current credit application; and
    - the gender of the client.
    """)

    st.markdown("""
    # *Left hand side graph interpretation*

    Colors indicate whether the variable is positively or negatively affecting the outcome of a client's credit application as its value increases. Blue color towards the right hand side of the graph indicates a positive influence. The higher the credit rating and the down payment, the higher the likelihood of a client's application being accepted; a history of underpayments, a relatively large number of other existing credits, or the male gender of the applicant all negatively affect the outcome of the application (red towards the right hand side of the graph).
    """)