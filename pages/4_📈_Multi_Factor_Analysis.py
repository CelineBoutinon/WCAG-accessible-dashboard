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
# colorblind_palette = px.colors.qualitative.Safe  # colorblind-friendly palette
# default_palette = px.colors.qualitative.Plotly
# palette = colorblind_palette if accessibility_mode else default_palette

# Get selected client application id
selected_value = st.session_state.get("selected_value", None)

HOME_PAGE = "streamlit_cloud_app_P8_v1.py"  # or "Home.py" or the correct filename

if st.session_state.accessibility_mode:
    if selected_value is None:
        st.warning("# Please select a client credit application reference on the Home page first.")
        st.markdown("# 🏠︎ [Back to Home](streamlit_cloud_app_P8_v1.py)")
        st.page_link(HOME_PAGE, label="Back to Home")
    else:
        st.write(f"# Using selected client application: {selected_value}")
        st.write("# 🗣 Accessibility mode is ON - navigate to home page to turn off")
        st.markdown("# 🏠︎ [Back to Home](streamlit_cloud_app_P8_v1.py)")
        st.page_link(HOME_PAGE, label="Back to Home")
else:
    if selected_value is None:
        st.warning("### Please select a client credit application reference on the Home page first.")
        st.page_link(HOME_PAGE, label="🏠 Back to Home")
    else:
        st.write(f"### Using selected client application: {selected_value}")
        st.write("### 🛈 Accessibility mode is OFF - navigate to home page to turn on")
        st.page_link(HOME_PAGE, label="🏠 Back to Home")

if st.session_state.accessibility_mode:
    st.image("bandeau_bw.png", caption="Company Logo: Prêt à Dépenser", use_container_width=True)  # Added alt text via caption
else:
    st.image("bandeau.png", use_container_width=False)

# Get selected client application id
# selected_value = st.session_state.get("selected_value", None)
# if selected_value is None:
    # st.warning("Please select a client credit application reference on the Home page first.")
# else:
    # st.write(f"Using selected client application: {selected_value}")

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

# Get selected client application id
# selected_value = st.session_state.get("selected_value", None)
# if selected_value is None:
    # st.warning("Please select a client credit application reference on the Home page first.")
# else:
    # st.write(f"Using selected client application: {selected_value}")

# Check if API data is loaded
    # if "app_data" not in st.session_state:
        # st.warning("API data not loaded yet. Please go back to Home page and select a client.")
    # else:
       # Access stored data
        # shap_values_client_dict = st.session_state.shap_values_client_dict
        # shap_values_array = st.session_state.shap_values_array
        # feature_names = st.session_state.feature_names
        # base_value = st.session_state.base_value
        # threshold_value = st.session_state.threshold_value
        # client_data_dict = st.session_state.client_data_dict
        # client_data_array = st.session_state.client_data_array
        # app_data = st.session_state.app_data
        # app_response = st.session_state.app_response

        # st.write(f"Using client application {selected_value}")
        # Example: show feature names and SHAP values
        # st.write("Feature names:", feature_names)
        # st.write("SHAP values:", shap_values_array)


if not st.session_state.accessibility_mode:
    st.write("## 🖖 Step 4 - Choose fields for client bivariate analysis display:")
  
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

else:
    st.write("# 🖖🏿 Step 4 - Choose fields for client bivariate analysis display:")
  
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


    # Column selection
    st.write("# Choose the horizontal axis for the scatter plot:")
    x_column = st.selectbox(" ", float_cols)
    st.write("# You chose variable", x_column, "for the horizontal axis.")
    st.write("# Choose the vertical axis for the scatter plot:")
    y_column = st.selectbox("  ", float_cols)
    st.write("# You chose variable", y_column, "for the vertical axis.")
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
                        color_discrete_sequence=['dimgrey'],
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
                                    marker=dict(color='#242164', symbol='star', size=30),
                                    name=f'Client ID: {selected_value}'
                                    ))
        else:
            st.warning(f"Client ID {selected_value} not found in the dataset for the scatter plot.")

        st.plotly_chart(fig)
        st.write(f"# Scatter plot of clients according to {x_column} and {y_column} variables.")
        st.write(f"# Client number {selected_value} shown as star marker.")
        st.write("# Graph is interactive. Hover over top right corner of graph for zoom options.")

