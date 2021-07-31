import pandas as pd
import streamlit as st
from sklearn import preprocessing
from inference import predict

st.title('Diabetes Prediction App')

model = 'Random Forest'

col_names = ['Age', 'Sex', 'BMI', 'BP', 's1', 's2','s3', 's4', 's5', 's6']
result = []

form = st.form(key='columns_in_form')
cols = form.beta_columns(10)

for i, col in enumerate(cols):
    res = form.number_input(col_names[i], value=1, key=str(i))
    result.append(res)
    
submitted = form.form_submit_button('Submit')

if submitted:
    df = pd.DataFrame([result], columns=['Age', 'Sex', 'BMI', 'BP', 's1', 's2', 's3', 's4', 's5', 's6'])

    # Normalizing before sending it to prediction
    df = preprocessing.normalize(df)
    st.write('Selected  :', model)

    predictions = predict(df, model)
    form.success(f"Result: {predictions}")

#  Upload data for all the patients
csv_file = st.file_uploader('Choose a CSV file')
if csv_file:
    st.write('filename : ', csv_file.name)
    patient_diabetes_informations_df = pd.read_csv(csv_file, index_col=0)
    print(patient_diabetes_informations_df)
    patient_diabetes_informations_df = patient_diabetes_informations_df.drop(['target'], axis=1)
    st.write(patient_diabetes_informations_df)

# execute model
if st.button('Predict diabetes progression'):
    if csv_file is not None:
        predictions = predict(patient_diabetes_informations_df, model)
        st.write('Selected  :', model)
        st.success(f'Prediction:\n {predictions}')
    else:
        st.warning('You need to upload a csv file before')
