# LinkedIn User Prediction App

This Streamlit app uses a logistic regression model to predict whether a person is a LinkedIn user based on a set of demographic and socio-economic features including income level, education level, whethere a person is married, gender, whether the person is a parent, and their age.

## Features

- **Income Level (1-9)**: Income category from less than $10,000 to $150,000 or more.
- **Education Level (1-8)**: Education levels ranging from no formal schooling to postgraduate degree.
- **Parent**: Whether the person is a parent of a child under 18 years old.
- **Marital Status**: Whether the person is married.
- **Gender**: Gender of the individual.
- **Age**: The person's age, between 18 and 100.

The app uses a logistic regression model trained on a dataset containing user characteristics. The model is trained to classify whether an individual is likely to be a LinkedIn user (`sm_li` = 1) or not (`sm_li` = 0).

## How to Use the App

1. Choose the features (income, education, parent, marital status, gender, age).
2. Click on the "Predict" button to get the prediction.
3. The model will output whether the person is predicted to be a LinkedIn user, along with the probability.
