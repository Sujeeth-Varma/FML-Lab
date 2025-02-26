# Install the pgmpy library if it's not already installed in google collab
# !pip install pgmpy

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Load dataset
data = pd.read_csv("./datasets/6.csv")
heart_disease = pd.DataFrame(data)
print(heart_disease.head())  # Print only the first few rows for better readability

# Define Bayesian Model
model = BayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholesterol'),
    ('Lifestyle', 'diet'),
    ('cholesterol', 'heartdisease')
])

# Fit model using Maximum Likelihood Estimation
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Perform inference
HeartDisease_infer = VariableElimination(model)

# Input instructions
print('\nInput Options:')
print('For Age enter: SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4')
print('For Gender enter: Male:0, Female:1')
print('For Family History enter: Yes:1, No:0')
print('For Diet enter: High:0, Medium:1')
print('For Lifestyle enter: Athlete:0, Active:1, Moderate:2, Sedentary:3')
print('For Cholesterol enter: High:0, BorderLine:1, Normal:2')

# Take user input
try:
    evidence = {
        'age': int(input('Enter Age: ')),
        'Gender': int(input('Enter Gender: ')),
        'Family': int(input('Enter Family History: ')),
        'diet': int(input('Enter Diet: ')),
        'Lifestyle': int(input('Enter Lifestyle: ')),
        'cholesterol': int(input('Enter Cholesterol: '))
    }
    
    # Perform query
    q = HeartDisease_infer.query(variables=['heartdisease'], evidence=evidence)
    print(q)
except ValueError:
    print('Invalid input! Please enter numeric values according to the provided options.')
