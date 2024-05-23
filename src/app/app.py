import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# Função para carregar o modelo
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError as e:
        st.error(f"File not found")
        st.stop()

# Função para transformar os dados com o scaler
def transform_scaler(data):
    transformer = RobustScaler().fit(data[['Age','Fare']])
    return transformer.transform(data[['Age','Fare']])

# Carregue seu modelo (ajuste o caminho conforme necessário)
model_path = '../../saved_models/model_rf.pkl'
model = load_model(model_path)

# Título do aplicativo
st.title("Data Entry for ML Model Evaluation")

# Legendas e campos de entrada
st.header("Passenger Information")

# Passenger class
pclass = st.selectbox(
    "Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)",
    options=[1, 2, 3],
    help="Select the passenger class. Options are 1st, 2nd, or 3rd class."
)

# Passenger age
age = st.slider(
    "Age",
    min_value=1,
    max_value=90,
    help="Enter the passenger's age. Range is from 1 to 90."
)

# Siblings/Spouses aboard
sibsp = st.selectbox(
    "Does the passenger have siblings/spouses aboard? (Enter 0 for No, 1 for Yes)",
    options=[0, 1],
    help="Enter 0 if no siblings/spouses aboard, 1 if there are."
)

# Parents/Children aboard
parch = st.selectbox(
    "Does the passenger have parents/children aboard? (Enter 0 for No, 1 for Yes)",
    options=[0, 1],
    help="Enter 0 if no parents/children aboard, 1 if there are."
)

# Fare paid by the passenger
fare = st.number_input(
    "Fare",
    min_value=10.0,
    help="Enter the fare paid by the passenger. It should be a numerical value."
)

# Passenger gender
is_male = st.selectbox(
    "Is the passenger male? (Enter 1 for Yes, 0 for No)",
    options=[0, 1],
    help="Enter 1 if the passenger is male, 0 if female."
)

# Relatives onboard
relatives = st.selectbox(
    "Any relatives onboard? (Enter 0 for No, 1 for Yes)",
    options=[0, 1],
    help="Enter 1 if the passenger has relatives onboard, 0 if not."
)

# Port of embarkation
embarked = st.selectbox(
    "Port of Embarkation (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)",
    options=[0, 1, 2],
    help="Select the port of embarkation. 0 for Cherbourg, 1 for Queenstown, 2 for Southampton."
)

# Button to submit the data
if st.button("Submit"):
    # Check if fare is not null
    if fare is None or fare <= 0:
        st.error("Fare cannot be null or less than 0. Please enter a valid fare.")
    else:
        # Create a dictionary with the data
        data = {
            'Pclass': [pclass],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'IsMale': [is_male],
            'Relatives': [relatives],
            'Embarked': [embarked]
        }

        # Convert the dictionary to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Apply the scaler to the Age and Fare columns
        df[['Age', 'Fare']] = transform_scaler(df)
        
        # Make a prediction using the model
        prediction = model.predict(df)
        
        st.write("Result:")
        if prediction == 1:
            
            st.write("The passenger made it out of the sinking ship alive")    
        else:
            st.write("The passenger tragically died")