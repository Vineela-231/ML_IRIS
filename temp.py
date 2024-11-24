import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model=joblib.load('model.joblib')
df=np.array([[5.3,2.2,3,4],[2.2,1.1,4.2,5.6],[3.2,4.2,5.3,2.3]])

pred=model.predict_proba(df)
print(pred)