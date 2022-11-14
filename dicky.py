import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

st.title('Aplikasi Data Mining - Klasifikasi Menggunakan Naive Bayes')
st.write("Dataset Yang digunakan adalah..")

# dataset = st.sidebar.write('Dataset Iris Sepal')

st.sidebar.header("Parameter Inputan")


def inputUser():
    panjangSepal = st.sidebar.text_input("Panjang Sepal", 1.0)
    lebarSepal = st.sidebar.text_input("Lebar Sepal", 1.0)
    panjangPetal = st.sidebar.text_input("Panjang Petal", 1.0)
    lebarPetal = st.sidebar.text_input("Lebar Petal", 1.0)
    # panjangSepal = st.sidebar.slider("Panjang sepal", 1.0, 11.0, 10.0)
    # lebarSepal = st.sidebar.slider("Lebar sepal", 2.0, 7.0, 2.4)
    # panjangPetal = st.sidebar.slider("Panjang petal", 1.0, 5.0, 2.5)
    # lebarPetal = st.sidebar.slider("Lebar petal", 0.1, 3.0, 1.5)
    data = {
        # 'tes' : tes,
        'Panjang sepal': panjangSepal,
        'Lebar sepal': lebarSepal,
        'Panjang petal': panjangPetal,
        'Lebar petal': lebarPetal
    }
    fitur = pd.DataFrame(data, index=[0])
    return fitur


df = inputUser()

st.subheader('Parameter Inputan')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

model = GaussianNB()
model.fit(X, Y)

prediksi = model.predict(df)
prediksiProb = model.predict_proba(df)

st.subheader("Label kelas dan Nomor Indeks")
st.write(iris.target_names)

st.subheader("prediksi hasil klasifikasi")
st.write(iris.target_names[prediksi])

st.subheader("Prediksi proba")
st.write(prediksiProb)
