from unittest import result

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
from PIL import Image
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_heart = pickle.load(open(os.path.join(BASE_DIR, 'Model/Hasilgenerate.pkl'), 'rb'))

st.set_page_config(page_title="Health Prediction App", page_icon="❤️", layout="centered")

def predict_heart_disease():
    st.title("Heart Disease Prediction ")
    st.write("""SS
    This app predicts the likelihood of heart disease based various health parameters. Please fill in the details below to get your prediction. 
    Data obtained from the UCI Machine Learning Repository: [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
                """)

    st.image("public/heart-disease.jpg", caption="Heart Disease Prediction", use_container_width=True)

    st.sidebar.header("Input Parameters:")
    st.sidebar.markdown("""Please enter the following health parameters to predict the likelihood of heart disease.
    - Age: Age of the patient in years.
    - Sex: Gender of the patient (1= male, 2=female)
    - Chest Pain Type: Type of chest pain experienced by the patient (1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic)
    - thalium stress test (1=normal, 2=fixed defect, 3=reversible defect)
    - Exercise induced angina (1=yes, 0=no)
    - oldpeak: ST depression induced by exercise relative to rest
    - slope: the slope of the peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)
    - Number of major vessels (0-3) colored by flourosopy
    - Maximum heart rate achieved: Maximum heart rate achieved during the test            
                        """)

    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3,])
    if cp == 0:
        st.sidebar.write("Typical Angina")
    elif cp == 1:
        st.sidebar.write("Atypical Angina")
    elif cp == 2:
        st.sidebar.write("Non-Anginal Pain")
    elif cp == 3:
        st.sidebar.write("Asymptomatic")
    thal = st.sidebar.selectbox("Thalium Stress Test", [ 1, 2, 3])
    if thal == 1:
        st.sidebar.write("Normal")
    elif thal == 2:
        st.sidebar.write("Fixed Defect")
    elif thal == 3:
        st.sidebar.write("Reversible Defect")
    thalac = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    if thalac < 100:
        st.sidebar.write("Low Heart Rate")
    elif thalac < 150:
        st.sidebar.write("Normal Heart Rate")
    elif thalac < 200:
        st.sidebar.write("High Heart Rate")
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2,])
    if slope == 0:
        st.sidebar.write("Upsloping")
    elif slope == 1:
        st.sidebar.write("Flat")
    elif slope == 2:
        st.sidebar.write("Downsloping")
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.0, 1.0)
    if oldpeak < 1.0:
        st.sidebar.write("Normal ST Depression")
    elif oldpeak <= 2.0:
        st.sidebar.write("Mild ST Depression")
    elif oldpeak > 2.0:
        st.sidebar.write("Moderate ST Depression")
    else:
        st.sidebar.write("Severe ST Depression")
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    if exang == 0:
        st.sidebar.write("No Exercise Induced Angina")
    elif exang == 1:
        st.sidebar.write("Exercise Induced Angina")
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Flourosopy", [0, 1, 2, 3])
    if ca == 0:
        st.sidebar.write("No Major Vessels Colored")
    elif ca == 1:
        st.sidebar.write("One Major Vessel Colored")
    elif ca == 2:
        st.sidebar.write("Two Major Vessels Colored")
    elif ca == 3:
        st.sidebar.write("Three Major Vessels Colored")
    sex = st.sidebar.selectbox("Sex", [1, 2])
    if sex  == 1:
        st.sidebar.write("Male")
    elif sex == 2:
        st.sidebar.write("Female")
    age = st.sidebar.slider("Age", 20, 100, 50)
    if age < 30:
        st.sidebar.write("Young Adult")
    elif age <= 60:  
        st.sidebar.write("Middle Aged Adult")
    elif age >= 70:
        st.sidebar.write("Older Adult")

    data = {
        'cp': cp,
        'thalach': thalac,
        'slope': slope,
        'oldpeak': oldpeak,
        'exang': exang,
        'ca': ca,
        'thal': thal,
        'sex': sex,
        'age': age
    }
    input_data = pd.DataFrame([data])
    st.write(input_data)

    if st.sidebar.button("Predict"):
        with st.spinner("Predicting..."):
            time.sleep(2)
            prediction = model_heart.predict(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you are at risk of heart disease. Please consult a healthcare professional for further evaluation.")
            else:
                st.success("The model predicts that you are not at risk of heart disease. Keep up with a healthy lifestyle!")

with open("Model/iris.pkl", 'rb') as file:  
 loaded_model = pickle.load(file)

st.set_page_config(page_title="Halaman Modelling", layout="wide")


def iris():
    st.write("""
        This app predicts the **Iris Species**
        
        Data obtained from the [iris dataset](https://www.kaggle.com/uciml/iris) by UCIML. 
        """)

    img = Image.open(os.path.join(BASE_DIR, "public/iris.jpg"))
    st.image(img, width=500)

    st.sidebar.header('Input Manual')
    SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', 4.3, 10.0, 5.0)
    SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', 2.0, 5.0, 3.3)
    PetalLengthCm = st.sidebar.slider('Petal Length (cm)', 1.0, 9.0, 4.5)
    PetalWidthCm = st.sidebar.slider('Petal Width (cm)', 0.1, 5.0, 1.4)

    data = {
        'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm
    }

    # ✅ Selalu buat input_df dari manual input dulu
    input_df = pd.DataFrame(data, index=[0])

    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # ✅ Kalau ada file upload, timpa input_df dengan data CSV
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

    st.write("User Input Features:")
    st.write(input_df)

    if st.sidebar.button('Predict!'):
        # ✅ Sekarang input_df selalu ada, tidak akan error
        loaded_model = pickle.load(open(os.path.join(BASE_DIR, 'model_iris.pkl'), 'rb'))
        prediction = loaded_model.predict(input_df)
        st.subheader('Prediction:')
        if prediction[0] == 0:
            st.success("🌸 Iris-setosa")          # ✅ Hijau
        elif prediction[0] == 1:
            st.warning("🌼 Iris-versicolor")      # ⚠️ Kuning
        else:
            st.error("🌺 Iris-virginica")         # ❌ Merah
        output = str(prediction[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")      
    



def about():
    st.title("About this App")
    st.write("""
    # Welcome to my machine learning dashboard

    This dashboard created by : [@nafiis-farhan](https://www.linkedin.com/in/https://www.linkedin.com/in/nafiis-farhan-0103b4294/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BKjedXTtVRSKMjZ0IyZv1zQ%3D%3D/)
             

    This app is a machine learning-based heart disease prediction tool. It uses a trained model to predict the likelihood of heart disease based on various health parameters. The model was trained on the UCI Heart Disease dataset and can provide insights into your heart health.
    **Disclaimer:** This app is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

    """)

def developer_info():
    st.title("Developer Information")
    st.write("""
    This app was developed by [Your Name]. You can reach out to me for any queries or collaborations. 
    this app is built using Streamlit, a powerful framework for creating interactive web applications with Python. 
    its for educational purposes and to demonstrate the capabilities of machine learning in healthcare.
    thank you for using this app and I hope it provides valuable insights into heart disease prediction.
    """)

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Heart Disease Prediction", "Iris Species Prediction", "About this App", "Developer Information"])
    
    if app_mode == "Heart Disease Prediction":
        predict_heart_disease()
    elif app_mode == "Iris Species Prediction":
        iris()
    elif app_mode == "About this App":
        about()
    elif app_mode == "Developer Information":
        developer_info()


if __name__ == "__main__":
    main()
