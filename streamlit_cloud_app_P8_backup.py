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
selected_value = st.number_input("", min_value=1, max_value=48745)

# Display the selected client credit application reference
st.write(f"### You selected client application: {selected_value}")
st.write("")
st.write("")


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
    st.write("")
    st.write("")


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

else:
    st.error(f"Failed to fetch data. API status code : {app_response.status_code}")


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


# Create scatter plot
client_data=pd.read_csv('test_data_final.csv').drop(labels=['Unnamed: 0'], axis=1)
column_names = client_data.columns.tolist()
cat_cols = ['NAME_CONTRACT_TYPE', 'INCOME_TYPE', 'EMPLOYMENT_SECTOR']
bool_cols = ['IS_MALE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_EMAIL', 'WHITE_COLLAR', 'UPPER_EDUCATION', 
             'IS_MARRIED', 'LIVES_INDEPENDENTLY', 'PHONE_PROVIDED', 'PHONE_REACHABLE', 'ADDRESS_MISMATCH']
int_cols = ['DAYS_ID_PUBLISH', 'REGION_RATING_CLIENT_W_CITY', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
            'DAYS_LAST_PHONE_CHANGE', 'REQUESTS_ABOUT_CLIENT_1Y', 'pcb_CNT_INSTALMENT_FUTURE_max', 
            'pa_NFLAG_INSURED_ON_APPROVAL_sum', 'pa_CREDIT_SECURITY_sum', 'TOTAL_APPROVED_CREDITS', 'TOTAL_ACTIVE_CAR_LOANS',
            'TOTAL_ACTIVE_OTHER_LOANS', 'TOTAL_ACTIVE_CONSUMER_LOANS', 'TOTAL_ACTIVE_MICRO_LOANS', 'TOTAL_ACTIVE_MORTGAGES',
            'TOTAL_ACTIVE_CREDIT_CARDS', 'DEBT_RENEGOCIATIONS', 'CLIENT_BAD_CREDIT_HISTORY', 'CLIENT_FRAUD_FLAG',
            'CLIENT_WITHDRAWN_APPLICATIONS', 'YEAR_BIRTH']
float_cols = ['AMT_CREDIT', 'DISPOSABLE_INCOME', 'DISPOSABLE_INCOME_per_capita', 'YEARS_EMPLOYED_AS_ADULT_%',
              'CREDIT_RATING', 'NB_APPLICATION_DOCUMENTS_%', 'b_AMT_CREDIT_MAX_OVERDUE_max', 'b_AMT_CREDIT_SUM_OVERDUE_sum',
              'ccb_AMT_CREDIT_LIMIT_ACTUAL_mean', 'ccb_CNT_DRAWINGS_TOTAL_mean', 'ccb_CARD_OVERDRAWN_%_mean', 
              'ip_EARLY_PMT_mean', 'ip_AMT_OVERPAID_%_mean', 'ip_AMT_UNDERPAID_%_mean', 'pa_AMT_APPLICATION_mean', 
              'pa_RATE_DOWN_PAYMENT_mean', 'pa_REMAINING_CREDIT_DURATION_Y_mean', 'TOTAL_PAYMENT_DELAYS_DAYS', 
              'DOWN_PAYMENT_CURR_%', 'DEBT_RATE_INC_CURR_%', 'b_DAYS_CREDIT_CARD_max']


st.write("## 👌 Step 3 - Choose fields for client bivariate analysis display:")
  
# Column selection
st.write("### Choose the horizontal axis for the scatter plot:")
x_column = st.selectbox(" ", float_cols)
st.write("### Choose the vertical axis for the scatter plot:")
y_column = st.selectbox("  ", float_cols)
# facet_x = st.selectbox("Choose horizontal segmentation category:", cat_cols)
# facet_x = 'IS_MALE'
# facet_y = st.selectbox("Choose vertical segmentation category:", bool_cols)


# Fetch client data
# client_data_response = requests.get(f"http://127.0.0.1:5000/client_data/{selected_value}")
if app_response.status_code == 200:
    # client_data['IS_MALE'] = client_data['IS_MALE'].astype(str)
    # color_map = {'0': 'hotpink', '1': 'cornflowerblue'}
    # client_data['IS_MALE_COLOR'] = client_data['IS_MALE'].map(color_map)

    fig = px.scatter(client_data, x=x_column, y=y_column,
                     color_discrete_sequence=['#F1BD5F'],
                     # color='IS_MALE',
                     # color_discrete_map = {'0':'hotpink', '1':'cornflowerblue'},
                     trendline="ols", 
                     opacity=0.01,
                     # facet_col=facet_y,
                     # facet_row=facet_x, 
                     # labels={x_column: x_column, y_column: y_column}
                     )

    # Highlight trendline
    # fig.data[1].line.color = "fuchsia"

    # Highlight selected client on scatterplot
    if selected_value in client_data.index:
        selected_client = client_data.loc[[selected_value]]
        fig.add_trace(go.Scatter(x=selected_client[x_column],
                                 y=selected_client[y_column],
                                 mode='markers',
                                 marker=dict(color='#242164', symbol='star', size=15),
                                 name=f'Client ID: {selected_value}'
                                 ))
    else:
        st.warning(f"Client ID {selected_value} not found in the dataset for the scatter plot.")

    st.plotly_chart(fig)
    st.write("")
    st.write("")

    st.write("## 🖖 Step 4 - Choose field for client univariate analysis display:")
    # Create boxplot
    st.write("### Choose a column for the box plot:")
    box_column = st.selectbox("", float_cols)
    fig2 = go.Figure()
    fig2.add_trace(go.Violin(y=client_data[box_column],
                  name='Clients Distribution',
                  marker=dict(color='#F1BD5F'),
                  opacity=0.8))

    if selected_value in client_data.index:
        selected_client_value = client_data.loc[selected_value, box_column] 
        
        # add a second axis that overlays the existing one
        fig2.layout.xaxis2 = go.layout.XAxis(overlaying='x', range=[0, 2], showticklabels=False)
        fig2.add_scatter(x = [0, 2],
                         # y = [20, 20],
                         y=[selected_client_value,selected_client_value], # see https://stackoverflow.com/questions/58679441/python-plotly-add-horizontal-line-to-box-plot
                         mode='lines',
                         xaxis='x2',
                         showlegend=True,
                         name = f'Client ID: {selected_value}',
                         line=dict(dash='dot', color = "#f05876", width = 2))
        
        average_value = client_data[box_column].mean()
        # add a third axis that overlays the existing ones
        fig2.layout.xaxis3 = go.layout.XAxis(overlaying='x', range=[0, 2], showticklabels=False)
        fig2.add_scatter(x = [0, 2],
                         # y = [20, 20],
                         y=[average_value,average_value],
                         mode='lines',
                         xaxis='x3',
                         showlegend=True,
                         name = "Average - All clients",
                         line=dict(dash='solid', color = "#242164", width = 2))

    else:
        st.warning(f"Client ID {selected_value} not found in the dataset for the box plot.")

    # fig2.update_layout(title=f"Box plot of {box_column} with Selected Client",
    #                    yaxis_title=box_column)

    st.plotly_chart(fig2)


else:
    st.error(f"Failed to fetch Shap values for client application. API status code : {app_response.status_code}")


# Main function placeholder
def main():
    pass

if __name__ == "__main__":
    main()