# launch app locally by running flask --app app.py run --debug from the command line
# web app available at https://credit-scoring-api-0p1u.onrender.com

from flask import Flask, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load client test data
# client_data=pd.read_csv('X_test_final.csv')
client_data=pd.read_csv('test_data_final.csv')

# Load model
model = load('final_model.joblib')

# Load custom threshold
custom_threshold = load('optimal_threshold.joblib')

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Credit Scoring</h1>
<p>Welcome to the HOME CREDIT Credit Scoring app.</p>
<p>- use /client/ID to access client demographics</p>
<p>- use /predict/ID to retrieve client credit application decision</p>
<p>where ID is the client's unique Home Credit application number (whole number between 1 and 48745)</p>
'''

@app.route('/client/<int:id>', methods=['GET'])
def api_id(id):
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
    return jsonify('Client Home Credit application number:', id, 'Client summary information:', results)
    
@app.route("/predict/<int:id>", methods=['GET'])
def predict(id):
    # Ensure client id exists in test data
    if (id-1) >= client_data.shape[0]:            
        return "Error: Client id not in application database. Enter a whole number between 1 and 48745.", 404
    if ((id-1) < 0):
        return "Error: Client id not in application database. Enter a whole number between 1 and 48745.", 404
    # Load client data
    client_particulars = client_data.iloc[[id-1]]
    # Predict outcome of client credit application
    prediction = model.predict_proba(client_particulars) # model.predict(client_particulars) directly returns class 0 (no default)
    # or class 1 (default)
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
        'Client id': id,
        'Client default probability': proba, 
        'Class': proba_class,
        'Decision': decision,
        'Key Decision Factors': sorted_top_5_dict,
        'Expected Shap Value' : expected_value,
        'Shap values client' : shap_values_client.to_json(orient='records')
    })

if __name__ == "__main__":
    app.run()