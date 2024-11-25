import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('Iris model')
st.subheader("The model will predict the type of iris based on sepal length  and width")

df=np.array([[5.3,2.2,3,4],[2.2,1.1,4.2,5.6],[3.2,4.2,5.3,2.3]])

st.header('uploade data sample')
df
model=joblib.load('model.joblib')
pred=model.predict_proba(df)
pred=pd.DataFrame(pred,columns=['setosta','versi','virginica'])

st.header("predicted values")
pred
