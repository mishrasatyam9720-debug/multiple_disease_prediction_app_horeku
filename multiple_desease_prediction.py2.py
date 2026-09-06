import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


diabetes_model = pickle.load(
    open('classifier_model (1).sav', 'rb')
)

diabetes_model_scaler = pickle.load(
    open('scaler_model (1).sav', 'rb')
)

heart_disease_model = pickle.load(
    open('heart.model.sav', 'rb')
)

parkinsons_model = pickle.load(
    open('parkinsons_model.sav', 'rb')
)


# Sidebar for navigation
with st.sidebar:

    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction'],
        icons=['activity','heart','person'] ,
        default_index=0
    )


# Diabetes Prediction Page

if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI Value')

    with col1:
        DiabetesPedigreeFunction = st.text_input(
            'Diabetes Pedigree Function value'
        )

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''


    # creating a button for Prediction
    if st.button('Diabetes Test Result'):

        input_data = np.asarray([
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ])

        # reshape the array
        input_data_reshaped = input_data.reshape(1, -1)

        # standardizing the input data
        std_data = diabetes_model_scaler.transform(input_data_reshaped)

        # prediction
        prediction = diabetes_model.predict(std_data)

        if prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'


    st.success(diab_diagnosis)
    
# Heart Disease Prediction Page

if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain Type')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Cholesterol Level')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar')

    with col1:
        restecg = st.text_input('Resting ECG Results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('Oldpeak')

    with col2:
        slope = st.text_input('Slope')

    with col3:
        ca = st.text_input('Number of Major Vessels')

    with col1:
        thal = st.text_input('Thal')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):

        heart_prediction = heart_disease_model.predict([[
            float(age),
            float(sex),
            float(cp),
            float(trestbps),
            float(chol),
            float(fbs),
            float(restecg),
            float(thalach),
            float(exang),
            float(oldpeak),
            float(slope),
            float(ca),
            float(thal)
        ]])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person has heart disease'
        else:
            heart_diagnosis = 'The person does not have heart disease'

    st.success(heart_diagnosis)
# Parkinsons Prediction Page

if selected == 'Parkinsons Prediction':

    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col1:
        jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col2:
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')

    with col3:
        rap = st.text_input('MDVP:RAP')

    with col1:
        ppq = st.text_input('MDVP:PPQ')

    with col2:
        ddp = st.text_input('Jitter:DDP')

    with col3:
        shimmer = st.text_input('MDVP:Shimmer')

    with col1:
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')

    with col2:
        apq3 = st.text_input('Shimmer:APQ3')

    with col3:
        apq5 = st.text_input('Shimmer:APQ5')

    with col1:
        mdvp_apq = st.text_input('MDVP:APQ')

    with col2:
        dda = st.text_input('Shimmer:DDA')

    with col3:
        nhr = st.text_input('NHR')

    with col1:
        hnr = st.text_input('HNR')

    with col2:
        rpde = st.text_input('RPDE')

    with col3:
        dfa = st.text_input('DFA')

    with col1:
        spread1 = st.text_input('spread1')

    with col2:
        spread2 = st.text_input('spread2')

    with col3:
        d2 = st.text_input('D2')

    with col1:
        ppe = st.text_input('PPE')


    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):

        parkinsons_prediction = parkinsons_model.predict([[
            float(fo),
            float(fhi),
            float(flo),
            float(jitter_percent),
            float(jitter_abs),
            float(rap),
            float(ppq),
            float(ddp),
            float(shimmer),
            float(shimmer_db),
            float(apq3),
            float(apq5),
            float(mdvp_apq),
            float(dda),
            float(nhr),
            float(hnr),
            float(rpde),
            float(dfa),
            float(spread1),
            float(spread2),
            float(d2),
            float(ppe)
        ]])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
