# To run app locally, navigate to : C:\Users\celin\DS Projets Python\OCDS-repos-all\credit-scoring-api
# in the command line and run : py -m streamlit run streamlit_cloud_app_vf.py
# App runs on Streamlit Community Cloud at https://credit-scoring-api-f6g7yxlaqnt3qhu7rk7rkf.streamlit.app/
# IMPORTANT: In advanced settings, choose Python 3.10 when deploying the app in Streamlit Community Cloud
# to avoid errors related to distutils (discontinued from Python 3.12 onwards).

import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt
import numpy as np

# Display title & company logo
st.title("Welcome to the ")
st.image("logo.png")
st.title("Credit Scoring App!")

# Get user to select client credit application reference
# selected_value = st.select_slider("Select a client credit application reference:", options=range(1, 48745)) # slider selection
selected_value = st.number_input("Enter a client credit application reference:", min_value=1, max_value=48745,
                                 key="client_id") # direct client credit application reference input

# Display the selected client credit application reference
st.write(f"You selected client application: {selected_value}")

# Send a get request to the API using the selected client credit application reference
app_response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}")
app_data = app_response.json()  
shap_values_client_json = app_data["Shap values client"]
shap_values_client_dict = json.loads(shap_values_client_json)[0]
shap_values_array = np.array(list(shap_values_client_dict.values()))
feature_names = list(shap_values_client_dict.keys())
base_value = app_data.get("Expected Shap Value") 

# Display the response from the API (optional)
if app_response.status_code == 200:
    # st.write(f"App data: {app_data}")
    # st.write("Client id:", app_data['Client id'])  
    st.write(f"Client default probability: {app_data['Client default probability'] * 100:.2f}%")
    st.write("Class :", app_data['Class'])
    st.write("Decision :", app_data['Decision'])
else:
    st.error(f"Failed to fetch data. Status code : {app_response.status_code}")

# Create SHAP explanation object
if shap_values_array is not None:
    shap_explanation = shap.Explanation(values=shap_values_array, 
                                        base_values=base_value,
                                        feature_names=feature_names)
    # Create SHAP waterfall plot
    fig, ax = plt.subplots(figsize=(10,6))
    st.title(f"Key decision factors for client {app_data['Client id']}")
    shap.plots.waterfall(shap_explanation, max_display=6) # Show the top 5 features and group the remaining features
    st.pyplot(fig)
else: 
    st.error(f"Failed to fetch Shap values for client application. Status code : {app_response.status_code}")

# Main function placeholder
def main():
    pass

if __name__ == "__main__":
    main()