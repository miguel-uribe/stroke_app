Stroke Risk Evaluation Tool

This project is an interactive web application built with Plotly Dash that predicts a user's risk of stroke. It uses a machine learning model trained with the PyCaret library to provide a risk assessment based on various health and lifestyle factors.

Features

Interactive Web Interface: A user-friendly interface for inputting health data.
Machine Learning Integration: Utilizes a pre-trained PyCaret classification model to predict stroke risk.
Dynamic Risk Scoring: Calculates a risk score and categorizes it into levels (Very Low, Low, Medium, High) based on predefined quartiles.
Responsive Design: The user interface is designed to be functional across different screen sizes.

Project Structure

The repository is organized as follows:

├── assets/
│   └── style.css         # CSS for styling the web application
├── app.py                # The main Dash application script
├── stroke_risk_model.pkl # The trained PyCaret model pipeline
├── risk_levels.json      # JSON file defining risk categories and score limits
├── requirements.txt      # Python dependencies for the project
└── Procfile              # Command for web server (for deployment)

Setup and Installation

To run this application locally, you will need Python 3.8+ installed.

1. Clone the Repository:
git clone <your-repository-url>
cd <repository-name>
2. Create and Activate a Virtual Environment (Recommended):
Windows:python -m venv venv
.\venv\Scripts\activate
macOS/Linux:python3 -m venv venv
source venv/bin/activate
3. Install Dependencies:
Install all the required Python libraries using the requirements.txt file.
pip install -r requirements.txt

How to Run the Application
Once the setup is complete, you can start the application by running the app.py script from your terminal:
python app.py
The application will be available in your web browser at the following address:http://127.0.0.1:8050/

How It Works
The application follows a simple workflow:
Data Input: The user enters their information into the various dropdowns and input fields in the web interface.
Data Processing: When the "Calculate Risk" button is clicked, the inputs are collected into a Pandas DataFrame.
Prediction: The DataFrame is passed to the predict_model function from PyCaret. The loaded model pipeline automatically handles all necessary preprocessing steps (e.g., one-hot encoding, scaling) before making a prediction.
Risk Assessment: The model returns a prediction_score (the probability of stroke). This score is then compared against the intervals defined in risk_levels.json to assign a qualitative risk level ("Low", "Medium", "High", etc.).
Display Results: The calculated score and the corresponding risk level are displayed to the user.DeploymentThis application is ready for deployment on services like Render or Heroku. 
The Procfile and requirements.txt files are included for this purpose. The Procfile specifies the command to start a Gunicorn web server to run the application in a production environment.web: gunicorn app:server
