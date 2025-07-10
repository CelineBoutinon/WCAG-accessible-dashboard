# !!! THIS IS THE OCDS P8 APP !!!

# to run locally, navigate to C:\Users\celin\DS Projets Python\OCDS-repos-all\OCDS-P8-API
# and launch app by running flask --app app.py run --debug from the command line
# web app available at https://credit-scoring-api-0p1u.onrender.com

from flask import Flask, jsonify
from joblib import load
import pandas as pd
import streamlit as st

app = Flask(__name__)

# Set the page title for accessibility (WCAG 2.4.2)
st.set_page_config(page_title="Credit Scoring - Home", layout="wide")

# Accessibility mode
accessibility_mode = st.session_state.get("accessibility_mode", False)

if accessibility_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: black !important;
            color: white !important;
            font-size: 20px !important;
        }
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

# Load client test data
client_data=pd.read_csv('test_data_final.csv').drop(labels=['Unnamed: 0'], axis=1)

# Load model
model = load('final_model.joblib')

# Load custom threshold
custom_threshold = load('optimal_threshold.joblib')

# Add a descriptive alt text for the banner image (WCAG 1.1.1)
st.image("bandeau.png", caption="Logo Home Credit - Credit Scoring Application", use_column_width=True)


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Credit Scoring</h1>
    <p>Welcome to the HOME CREDIT Credit Scoring app.</p>
    <p>- use /predict/ID to retrieve client credit application decision</p>
    <p>where ID is the client's unique Home Credit application number (whole number between 1 and 48745)</p>'''

    st.title("Credit Scoring Application")
    st.markdown("""
    Welcome to the Home Credit Credit Scoring app.

    - Use the sidebar to navigate between pages.
    - Accessibility mode is available for users with visual impairments.
    """)

    # Example: Add a button to toggle accessibility mode
    if st.button("Toggle Accessibility Mode"):
        st.session_state["accessibility_mode"] = not accessibility_mode
        st.experimental_rerun()

    # Add instructions for keyboard navigation (optional, helps accessibility)
    st.markdown("""
    **Tip:** You can use the `Tab` key to navigate between interactive elements.
    """)


 
@app.route("/predict/<int:id>", methods=['GET'])
def predict(id):
    # Ensure client id exists in test data
    if (id-1) >= client_data.shape[0]:            
        return "Error: Client id not in application database. Enter a whole number between 1 and 48745.", 404
    if ((id-1) < 0):
        return "Error: Client id not in application database. Enter a whole number between 1 and 48745.", 404
    
    # Display summary client demographics
    result_cols = ['INCOME_TYPE', 'EMPLOYMENT_SECTOR', 'DISPOSABLE_INCOME_per_capita', 'YEAR_BIRTH', 'CREDIT_RATING',
                   'CLIENT_BAD_CREDIT_HISTORY', 'CLIENT_FRAUD_FLAG', 'IS_MALE', 'WHITE_COLLAR', 'UPPER_EDUCATION',
                   'IS_MARRIED', 'LIVES_INDEPENDENTLY']
    results = []
    row_data = client_data.loc[id-1, result_cols].to_dict()
    for k, v in row_data.items():
        if v==0:
            row_data[k] = 'no'
        if v==1:
            row_data[k] = 'yes'
    row_data['AGE'] = row_data.pop('YEAR_BIRTH')
    results.append(row_data)
    
    # Load client data
    client_particulars = client_data.iloc[[id-1]]
    # Predict outcome of client credit application
    # model.predict(client_particulars) directly returns class 0 (no default) or class 1 (default)
    prediction = model.predict_proba(client_particulars) 
    proba = prediction[0][1] # prediction[0][0] is proba of client NOT defaulting
    if proba > custom_threshold:
        proba_class = 'default'
        decision = "reject loan application"
    else:
        proba_class = 'no default'
        decision = "grant loan"

    # shap won't work with MLFlow pyfunc model => load pre-calculated Shap values for test data
    # shap_values_all = pd.DataFrame(load('shap_values_test.joblib'))
    shap_values_all = pd.read_csv('shap_values_test_data.csv')
    shap_values_client = shap_values_all.iloc[[id-1]]
    abs_values = shap_values_client.abs()
    expected_value = load('expected_value.joblib')

    # identify top 5 shap values for client prediction
    top_5_indices = abs_values.iloc[0].nlargest(5).index.values.tolist()
    top_5_columns = shap_values_client[top_5_indices].values.tolist()
    top_5_dict = {}
    for top_k, top_v in zip(top_5_indices, top_5_columns[0]):
        top_5_dict[top_k] = top_v
    sorted_top_5_dict = sorted(top_5_dict.items(), key=lambda top_5_dict: top_5_dict[1], reverse=True)
    
    # Return bank decision on client credit application
    return jsonify({
        'Client Home Credit application number:': id,
        'Client summary information' : results,
        'Client default probability': proba, 
        'Class': proba_class,
        'Decision': decision,
        'Key Decision Factors': sorted_top_5_dict,
        'Expected Shap Value' : expected_value,
        'Shap values client' : shap_values_client.to_json(orient='records'),
        'Client data' : client_particulars.to_json(orient='records'),
        'Threshold': custom_threshold
    })

if __name__ == "__main__":
    app.run()