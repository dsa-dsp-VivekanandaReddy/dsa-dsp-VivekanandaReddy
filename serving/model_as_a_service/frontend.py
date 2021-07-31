import pandas as pd
import streamlit as st
import requests
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder

server_url = 'http://127.0.0.1:8000'
predict_single_endpoint = '/predict_single'
predict_file_endpoint = '/predict'
get_endpoint = '/get'

st.title('Diabetes Prediction App')
model = 'Random Forest'


col_names = ['Age', 'Sex', 'BMI', 'BP', 's1', 's2', 's3', 's4', 's5', 's6']
result = []

form = st.form(key='columns_in_form')
cols = form.beta_columns(10)

for i, col in enumerate(cols):
    res = form.number_input(col_names[i], key=str(i))

    result.append(res)
submitted = form.form_submit_button('Submit')

if submitted:
    # Normalizing before sending it to prediction
    data = {}
    for i in range(len(col_names)):
        data[col_names[i].lower()] = result[i]

    json_object = json.dumps(data)

    ret = requests.post(server_url + predict_single_endpoint, data=json_object)
    res = json.loads(ret.content)
    form.success(res['result'])

#  Upload data for all the patients
csv_file = st.file_uploader('Choose a CSV file')
if csv_file:
    st.write('filename : ', csv_file.name)
    patient_diabetes_informations_df = pd.read_csv(csv_file)
    patient_diabetes_informations_df = patient_diabetes_informations_df.drop(['target'], axis=1)
    st.write(patient_diabetes_informations_df)

# execute model
if st.button('Predict diabetes progression'):
    if csv_file is not None:
        m = MultipartEncoder(fields={"file": ("filename", csv_file, "csv")})

        ret = requests.post(server_url+predict_file_endpoint, data=m,
                            headers={"Content-Type": m.content_type}, timeout=8000)

        res = json.loads(ret.content)
        st.success(res['result'])
    else:
        st.warning('upload a csv file')