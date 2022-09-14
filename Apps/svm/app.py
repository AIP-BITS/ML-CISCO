import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# from util import arr
from util import generate_data
import seaborn as sns
from sklearn.svm import SVC
from util import plot_decision_boundaries
import matplotlib.pyplot as plt

st.sidebar.subheader("Various Options")
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ("linear","iris", "moons", "circles")
)

sample_size = st.sidebar.slider('Sample Size', 100, 500, 100)
noise_level = st.sidebar.slider('Noise Level', 0.0, 1.0, 0.0)

kernel = st.sidebar.selectbox(
    "Select Kernel",
    ("linear", "poly", "rbf", "sigmoid")
)

degree = st.sidebar.slider('Poly Degree', 2, 15, 1)
clevel = st.sidebar.number_input('Enter C Level',min_value=0.00001)

gama = st.sidebar.selectbox(
    "Select Gamma Level",
    ("scale", "auto")
)
# gama = st.sidebar.number_input('Gamma Level', 0.00001, 1.0)
# st.write('The current number is ', gama)

X,y = generate_data(sample_size,dataset,noise_level)
## Train model

svm_clf = SVC(kernel=kernel,degree=degree,gamma=gama,C=clevel,random_state=42)
svm_clf.fit(X, y)

fig, ax = plt.subplots()
sns.scatterplot(x=X[:,0],y=X[:,1],hue=y)
plot_decision_boundaries(X,y,SVC,kernel=kernel,degree=degree,gamma=gama,C=clevel,random_state=42)
st.pyplot(fig)