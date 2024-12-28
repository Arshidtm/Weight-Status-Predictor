import numpy as np
import  streamlit as st
import pickle as pkl

model=pkl.load(open('model_RandomForest.pkl','rb'))
model2=pkl.load(open('model_KNN.pkl','rb'))
model3=pkl.load(open('model_RFpipe.pkl','rb'))

st.title("Weight Status Predictor")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (in years)", min_value=0, max_value=120, step=1)
height = st.number_input("Height (in cm)", min_value=50.0, max_value=300.0, step=0.01)
weight = st.number_input("Weight (in kilograms)", min_value=0.0, max_value=300.0, step=0.1)

# Calculate BMI

if height > 0:
    height_in_meters = height / 100
    bmi = weight / (height_in_meters ** 2)

labels={'Normal Weight': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3}
input=[[gender,age,height,weight,bmi]]

def pred(arr):
    val=model3.predict(arr)
    for key,vals in labels.items():
        if val[0]==vals:
            return key
result=pred(input)

if st.button("Predict Weight Status"):
    st.write(result)

