from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import streamlit as st
import  numpy as np
import tensorflow as tf
import pickle
import pandas as pd


model=tf.keras.models.load_model('model.h5')


with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)


with open('onehotencoder.pkl','rb') as file:
    onehotencoder=pickle.load(file)


with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

st.title('Customer Churn  Prediction')


geography=st.selectbox('Geography',onehotencoder.categories_[0])
gender=st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])






input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded=onehotencoder.transform(pd.DataFrame({'Geography': [geography]}))
geo_encoded_df=pd.DataFrame(geo_encoded.toarray(),columns=onehotencoder.get_feature_names_out(['Geography']))


input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data=scaler.transform(input_data)


prediction=model.predict(input_data)
prediction_proba=prediction[0][0]

st.write(f'Churn Probability:{prediction_proba:.2f}')

if prediction_proba>0.5:
    st.write('The customer is likely to Churn')
else:
    st.write('Not likely TO Churn')