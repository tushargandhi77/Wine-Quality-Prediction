import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
df = pickle.load(open('5_best.pkl','rb'))

st.title('Wine Quality Predictor')

alcohol = st.selectbox('Alcohol',df['alcohol'].unique())

sulphate = st.selectbox('Sulphate',df['sulphates'].unique())

volatile = st.selectbox('Volatile Acidity',df['volatile acidity'].unique())

tsd = st.selectbox('Sulphur dioxide',df['total sulfur dioxide'].unique())

density = st.selectbox('Density',df['density'].unique())

if st.button('Predict Quality'):
    query = np.array([alcohol,sulphate,volatile,tsd,density])
    st.title("The Predicted Quality: "+str(model.predict([query])))