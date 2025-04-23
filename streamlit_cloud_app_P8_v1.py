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
# import streamviz

# Display title & company logo
st.title("Welcome to the ")
st.image("logo.png")
st.title("Credit Scoring App!")

# Get user to select client credit application reference
# selected_value = st.select_slider("Select a client credit application reference:", options=range(1, 48745)) # slider selection
selected_value = st.number_input("Enter a client credit application reference:", min_value=1, max_value=48745) # direct client credit application reference input

# Display the selected client credit application reference
st.write(f"You selected client application: {selected_value}")

# Send a get request to the API using the selected client credit application reference
# app_response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}")
app_response = requests.get(f"http://127.0.0.1:5000/predict/{selected_value}")
app_data = app_response.json()  
shap_values_client_json = app_data["Shap values client"]
shap_values_client_dict = json.loads(shap_values_client_json)[0]
shap_values_array = np.array(list(shap_values_client_dict.values()))
feature_names = list(shap_values_client_dict.keys())
base_value = app_data.get("Expected Shap Value")
threshold_value =  app_data.get("Threshold")

# Display the response from the API (optional)
if app_response.status_code == 200:
    # st.write(f"App data: {app_data}")
    # st.write("Client id:", app_data['Client id'])  
    st.write(f"Client default probability: {app_data['Client default probability'] * 100:.2f}%")
    st.write("Class :", app_data['Class'])
    st.write("Decision :", app_data['Decision'])
else:
    st.error(f"Failed to fetch data. Status code : {app_response.status_code}")


# Display client default probability on a gauge
bar_color = 'red' if app_data['Client default probability'] > threshold_value else 'forestgreen'

fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = app_data['Client default probability'] * 100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "forestgreen"}},
    title = {'text': "Client default probability",  'font': {'size': 24}},
    gauge = {'axis': {'range': [None, 100]},
             'bar': {'color': bar_color},
             'steps' : [
                 {'range': [0, 50], 'color': "palegreen"},
                 {'range': [50, 100], 'color': "lightcoral"}],
                  'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.502, 'value': 50}
                  }
    ))

st.plotly_chart(fig)


# Create SHAP explanation object
if shap_values_array is not None:
    shap_explanation = shap.Explanation(values=shap_values_array, 
                                        base_values=base_value,
                                        feature_names=feature_names)
    # Create SHAP waterfall plot
    fig, ax = plt.subplots(figsize=(10,6))
    st.title(f"Key decision factors for client {selected_value}")
    shap.plots.waterfall(shap_explanation, max_display=6) # Show the top 5 features and group the remaining features
    st.pyplot(fig)
else: 
    st.error(f"Failed to fetch Shap values for client application. Status code : {app_response.status_code}")

# Main function placeholder
def main():
    pass

if __name__ == "__main__":
    main()