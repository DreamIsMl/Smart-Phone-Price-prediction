import pickle

import numpy as np
import streamlit as st

st.title('Smart Phone Price Prediction')

with open('F:\\class\\Machine Learning\\project\\smart phone price prediction\\xtrain.pkl','rb') as data_file:
    data = pickle.load(data_file)

with open('F:\\class\\Machine Learning\\project\\smart phone price prediction\\smodel.pkl','rb') as model_file:
    model = pickle.load(model_file)
Company = st.selectbox("Selec Company Name",data['product'].unique())

ram = st.selectbox('Enter Ram Size',data['ram'].unique())
rom = st.selectbox("Select Rom Size",data['rom'].unique())
memeory = st.selectbox('Enter Memory Size',data['expandable memory'].unique())
camera= st.selectbox("Select camera",data['camera'].unique())
front_camera = st.selectbox('Enter Front Camera',data['front camera'].unique())
rating = st.selectbox("Enter Rating Of Your Phone",data['rating'].unique())

if st.button('Predict'):
    query= np.array([Company,ram,rom,memeory,camera,front_camera,rating])
    query = query.reshape(1,7)
    st.title("The prediction price is: " + str(int(np.exp(model.predict(query)[0]))))