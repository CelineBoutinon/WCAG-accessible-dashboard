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

