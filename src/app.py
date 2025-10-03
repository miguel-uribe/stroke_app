import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import json
import pickle
import pandas as pd
from pycaret.classification import load_model, predict_model


# --- Load Model and Risk Data ---
# Note: Ensure these files are in the correct directory relative to app.py
# The model should be in a 'content' folder at the same level as the app's parent directory.
# The risk levels json should be in a 'contents' folder at the same level.
try:
    # Use PyCaret's load_model to load the entire pipeline
    model = load_model('../content/stroke_risk_model')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Error: 'stroke_risk_model.pkl' not found or is invalid. Make sure the model file is in the correct path.")
    model = None

try:
    with open('../content/quartile_data.json', 'r') as f:
        risk_data = json.load(f)
except FileNotFoundError:
    print("Error: 'risk_levels.json' not found. Make sure the risk definition file is in the correct path.")
    risk_data = None


# --- Initialize the Dash App ---
app = dash.Dash(__name__)
server = app.server

# --- App Layout ---
app.layout = html.Div(className='container', children=[
    # In-memory storage components
    dcc.Store(id='work-type-store'),
    dcc.Store(id='smoking-store'),
    dcc.Store(id='hypertension-store'),
    dcc.Store(id='heart-disease-store'),
    dcc.Store(id='residence-store'),
    dcc.Store(id='age-store'),
    dcc.Store(id='glucose-store'),
    dcc.Store(id='bmi-store'),
    dcc.Store(id='gender-store'),
    dcc.Store(id='married-store'),

    # --- Header ---
    html.H1("Stroke Risk Evaluation Tool", className='header'),

    # --- Top Row: Figures and Text ---
    html.Div(className='row', children=[
        html.Div(className='column', children=[
            dcc.Graph(id='figure1', figure=json.load(open("../content/score_dist.json"))),
            html.H3("Score Distribution"),
            html.P("The figure shows the distribution of the model score in the test set, the score clearly shows a different distribution for positive and negative cases of stroke. This evidences the capability of the model to differentiate between the two classes.")
        ]),
        html.Div(className='column', children=[
            dcc.Graph(id='figure2', figure=json.load(open("../content/coefficients.json"))),
            html.H3("Coefficients"),
            html.P("The best model was a Logistic Regression, the figure above shows the coefficients of the features. Features with positive coefficients increase the risk of stroke, while those with negative coefficients decrease it. For categorical features, the coefficients represent the impact of each category relative to a baseline.")
        ]),
        html.Div(className='column', children=[
            dcc.Graph(id='figure3', figure=json.load(open("../content/risk_levels.json"))),
            html.H3("Risk levels"),
            html.P("The model score was divided into 4 quartiles, each representing a different risk level. The figure above shows the actual stroke rate in each quartile for both the train train (used to define the quartiles) and the test (used to evaluate their robustness). The results evidence that the risk assessment is robust and that the model can effectively stratify patients into different risk levels.")
        ]),
    ]),

    # --- Evaluation Section ---
    html.H2("Let's Evaluate Your Risk", className='sub-header'),

    # --- Input Controls ---
    html.Div(className='controls-grid', children=[
        # Dropdowns
        dcc.Dropdown(id='work-type-dropdown', options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], placeholder="Work Type"),
        dcc.Dropdown(id='smoking-dropdown', options=['formerly smoked', 'never smoked', 'smokes', 'Unknown'], placeholder="Smoking Status"),
        dcc.Dropdown(id='hypertension-dropdown', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], placeholder="Hypertension"),
        dcc.Dropdown(id='heart-disease-dropdown', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], placeholder="Heart Disease"),
        dcc.Dropdown(id='residence-dropdown', options=['Urban', 'Rural'], placeholder="Residence Type"),
        dcc.Dropdown(id='gender-dropdown', options=['Male', 'Female', 'Other'], placeholder="Gender"),
        dcc.Dropdown(id='married-dropdown', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], placeholder="Ever Married"),

        # Numerical Inputs
        dcc.Input(id='age-input', type='number', placeholder='Age'),
        dcc.Input(id='glucose-input', type='number', placeholder='Average Glucose Level'),
        dcc.Input(id='bmi-input', type='number', placeholder='BMI'),
    ]),

 # --- Button and Output ---
    html.Button('Calculate Risk', id='run-model-button', className='run-button', n_clicks=0),
    html.Div(id='model-output', className='output-area')
])

# --- Callbacks to store input values in dcc.Store ---
@app.callback(
    [Output('work-type-store', 'data'),
     Output('smoking-store', 'data'),
     Output('hypertension-store', 'data'),
     Output('heart-disease-store', 'data'),
     Output('residence-store', 'data'),
     Output('age-store', 'data'),
     Output('glucose-store', 'data'),
     Output('bmi-store', 'data'),
     Output('gender-store', 'data'),
     Output('married-store', 'data')],
    [Input('work-type-dropdown', 'value'),
     Input('smoking-dropdown', 'value'),
     Input('hypertension-dropdown', 'value'),
     Input('heart-disease-dropdown', 'value'),
     Input('residence-dropdown', 'value'),
     Input('age-input', 'value'),
     Input('glucose-input', 'value'),
     Input('bmi-input', 'value'),
     Input('gender-dropdown', 'value'),
     Input('married-dropdown', 'value')]
)
def store_inputs(work_type, smoking, hypertension, heart_disease, residence, age, glucose, bmi, gender, married):
    return work_type, smoking, hypertension, heart_disease, residence, age, glucose, bmi, gender, married

# --- Main callback for the button ---
@app.callback(
    Output('model-output', 'children'),
    Input('run-model-button', 'n_clicks'),
    [State('age-input', 'value'),
     State('glucose-input', 'value'),
     State('bmi-input', 'value'),
     State('hypertension-dropdown', 'value'),
     State('heart-disease-dropdown', 'value'),
     State('gender-dropdown', 'value'),
     State('married-dropdown', 'value'),
     State('work-type-dropdown', 'value'),
     State('residence-dropdown', 'value'),
     State('smoking-dropdown', 'value')],
    prevent_initial_call=True
)
def run_model(n_clicks, age, glucose, bmi, hypertension, heart_disease, gender, married, work_type, residence, smoking):
    # Validate all inputs are different from None
    if None in [age, glucose, bmi, hypertension, heart_disease, gender, married, work_type, residence, smoking]:
        print (age)
        print (glucose)
        print (bmi)
        print (hypertension)
        print (heart_disease)
        print (gender)
        print (married)
        print (work_type)
        print (residence)
        print (smoking)
        print(all([age, glucose, bmi, hypertension, heart_disease, gender, married, work_type, residence, smoking]))
        return html.H3("Please fill in all the fields before calculating the risk.", style={'color': 'red'})

    if model is None or risk_data is None:
        return html.H3("Model or risk data is not loaded. Please check the server logs.", style={'color': 'red'})

    # Define the full feature set for the model
    model_features = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Male', 'gender_Other', 'ever_married_Yes',
        'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
        'smoking_status_formerly smoked', 'smoking_status_never smoked',
        'smoking_status_smokes'
    ]
    
    # Create a dictionary to hold the one-hot encoded data, initialized to 0
    patient_data = {feature: 0 for feature in model_features}

    # --- Populate the dictionary from user inputs ---
    # Numerical features
    patient_data['age'] = float(age)
    patient_data['avg_glucose_level'] = float(glucose)
    patient_data['bmi'] = float(bmi)
    patient_data['hypertension'] = 1 if hypertension == 'Yes' else 0
    patient_data['heart_disease'] = 1 if heart_disease == 'Yes' else 0
    
    # Categorical features
    if gender == 'Male': patient_data['gender_Male'] = 1
    elif gender == 'Other': patient_data['gender_Other'] = 1

    if married == 'Yes': patient_data['ever_married_Yes'] = 1

    if work_type == 'Never_worked': patient_data['work_type_Never_worked'] = 1
    elif work_type == 'Private': patient_data['work_type_Private'] = 1
    elif work_type == 'Self-employed': patient_data['work_type_Self-employed'] = 1
    elif work_type == 'children': patient_data['work_type_children'] = 1
    
    if residence == 'Urban': patient_data['Residence_type_Urban'] = 1

    if smoking == 'formerly smoked': patient_data['smoking_status_formerly smoked'] = 1
    elif smoking == 'never smoked': patient_data['smoking_status_never smoked'] = 1
    elif smoking == 'smokes': patient_data['smoking_status_smokes'] = 1
        

    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_data], columns=model_features)

    # --- Predict score and determine risk ---
    # Get the probability of the positive class (stroke=1)
    score = predict_model(model, data=patient_df, raw_score=True)['prediction_score_1'].values[0]
    print(score,)
    # Determine risk level based on the score
    risk_level = "Undefined"
    risk_percent = 0.0
    quartile_limits_str = risk_data['quartile_limits']
    stroke_rates = risk_data['test_stroke_rates']
    
    for i, limit_str in enumerate(quartile_limits_str):
        # Parse the interval string like "(-0.001, 0.005]"
        low_str, high_str = limit_str.strip('()[]').split(', ')
        low = float(low_str)
        high = float(high_str)
        
        if low < score <= high:
            risk_percent = stroke_rates[i] * 100
            if i == 0: risk_level = "Very Low"
            elif i == 1: risk_level = "Low"
            elif i == 2: risk_level = "Medium"
            elif i == 3: risk_level = "High"
            break

    return html.Div([
        html.H3("Risk Assessment Result:"),
        html.P(f"The model predicts a score of {score:.4f} for this profile."),
        html.P(f"Based on this score, the estimated risk level is: "),
        html.H4(f"{risk_level} ({risk_percent:.2f}%)", style={'color': '#007bff', 'font-weight': 'bold'})
    ])


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
