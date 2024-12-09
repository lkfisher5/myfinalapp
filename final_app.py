# Import packages
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset and preprocess it
def load_data():
    s = pd.read_csv('C:\Users\linds\OneDrive\Documents\Georgetown MSBA Classwork\OPAN-6607\Final Project\social_media_usage.csv')


    # Function clean_sm to clean the 'sm_li' column
    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    # Create a new dataframe called "ss" containing target and features
    ss = s.copy()

    # Apply the clean_sm function to create the 'sm_li' column (LinkedIn usage)
    ss['sm_li'] = clean_sm(ss['web1h'])

    # Clean and adjust other columns
    ss['income'] = np.where((ss['income'] >= 1) & (ss['income'] <= 9), ss['income'], np.nan)
    ss['educ2'] = np.where((ss['educ2'] >= 1) & (ss['educ2'] <= 8), ss['educ2'], np.nan)
    ss['par'] = np.where(ss['par'] == 1, 1, 0)
    ss['married'] = np.where(ss['marital'] == 1, 1, 0)
    ss['female'] = np.where(ss['gender'] == 2, 1, 0)
    ss['age'] = np.where(ss['age'] != 98, ss['age'], np.nan)

    # Drop rows with missing values
    ss = ss.dropna()

    # Define features (X) and target (y)
    X = ss[["income", "educ2", "par", "married", "female", "age"]]
    y = ss["sm_li"]
    
    return X, y

# Train the logistic regression model
def train_model():
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=123)
    
    model = LogisticRegression(class_weight='balanced', random_state=123)
    
    model.fit(X_train, y_train)
    
    return model

# Predict whether a person is a LinkedIn user or not, given their features
def predict(model, features):
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0][1]
    return prediction[0], probability

# Streamlit app
def main():
    # Add LinkedIn logo next to the header
    st.image("https://download.logo.wine/logo/LinkedIn/LinkedIn-Logo.wine.png", width=250)

    st.title("LinkedIn User Prediction")
    
    # Intro message
    st.markdown("""
    Welcome to the **LinkedIn User Prediction App**! This app uses a logistic regression model to predict whether an individual is a LinkedIn user based on a set of characteristics.  
    Simply choose the characteristics of a person, and the app will predict if they are likely to be a LinkedIn user, along with the probability of their usage.  
    """)
    
    # Input fields for features
    income = st.selectbox("Income Level (1-9)", [1, 2, 3, 4, 5, 6, 7, 8, 9])

    st.markdown("""
    **Income Level Breakdown:**
    - 1: Less than \$10,000
    - 2: \$10,000 to under \$20,000
    - 3: \$20,000 to under \$30,000
    - 4: \$30,000 to under \$40,000
    - 5: \$40,000 to under \$50,000
    - 6: \$50,000 to under \$75,000
    - 7: \$75,000 to under \$100,000
    - 8: \$100,000 to under \$150,000
    - 9: \$150,000 or more
    """, unsafe_allow_html=True)
    
    education = st.selectbox("Education Level (1-8)", [1, 2, 3, 4, 5, 6, 7, 8])
    
    st.markdown("""
    **Education Level Breakdown:**
    - 1: Less than high school (Grades 1-8 or no formal schooling)
    - 2: High school incomplete (Grades 9-11 or Grade 12 with NO diploma)
    - 3: High school graduate (Grade 12 with diploma or GED certificate)
    - 4: Some college, no degree (includes some community college)
    - 5: Two-year associate degree from a college or university
    - 6: Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)
    - 7: Some postgraduate or professional schooling, no postgraduate degree
    - 8: Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)
    """, unsafe_allow_html=True)
    
    parent = st.radio("Are you a parent of a child under 18?", ["Yes", "No"])
    marital_status = st.radio("Are you married?", ["Yes", "No"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=18, max_value=100)
    
    # Convert input to numeric values
    parent = 1 if parent == "Yes" else 0
    marital_status = 1 if marital_status == "Yes" else 0
    gender = 1 if gender == "Female" else 0
    
    features = [income, education, parent, marital_status, gender, age]
    
    if st.button("Predict"):
        model = train_model()  # Load and train the model
        
        # Get prediction and probability
        prediction, probability = predict(model, features)
        
        if prediction == 1:
            st.write("This person is classified as a LinkedIn user.")
        else:
            st.write("This person is classified as not a LinkedIn user.")
        
        st.write(f"Probability that this person is a LinkedIn user: {probability:.2f}")

if __name__ == "__main__":
    main()
