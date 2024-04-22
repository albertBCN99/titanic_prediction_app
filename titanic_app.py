import pandas as pd
import numpy as np
import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image



# Colocar la imagen en la parte superior
# Display the Titanic image

image = Image.open(r"titanic.png")
new_image = image.resize((600,150))

st.image(new_image)




st.write("""
# Titanic Survival Prediction App

This app predicts the **Survival** of the Titanic Passengers!
""")


         
st.sidebar.header('User Input Features')


# pclass	sex	age	siblings_spouses_count	embarked	survived


# Collects user input features into dataframe
def user_input_features():
    pclass = st.sidebar.selectbox('Class',(1,2,3))
    sex = st.sidebar.selectbox('Gender',('male','female'))
    age = st.sidebar.slider('Age', 0,80,30)
    siblings_spouses_count = st.sidebar.slider('Number of Siblings/Spouse onboard',0,8,1)
    embarked = st.sidebar.selectbox('Port Embarked',('Southampton','Cherbourg','Queenstown'))
    data = {'pclass':pclass,
            'sex' : sex,
            'age' : age,
            'siblings_spouses_count' : siblings_spouses_count,
            'embarked' : embarked}
    features = pd.DataFrame(data,index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
titanic_raw = pd.read_excel(r"C:\Users\alber\Desktop\CLASES\01_Streamlit\13_Titanic\titanic_streamlit.xlsx")
titanic = titanic_raw.drop(columns=['survived'])
df = pd.concat([input_df,titanic],axis=0)

# Encoding of ordinal features

encode = ['sex','embarked']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

print(type(df)) 


# Reads in saved classification model
load_clf = pickle.load(open(r"C:\Users\alber\Desktop\CLASES\01_Streamlit\13_Titanic\titanic_clf_2.pkl", 'rb'))

# Apply the model to make predictions on the user input data
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Map class labels for display
prediction_labels = ['Died', 'Survived']
predicted_class = prediction_labels[prediction[0]]

# Display the prediction result with emojis and larger font size
st.subheader('Prediction')

# Determine emoji and text based on predicted class
emoji = '☠️' if predicted_class == 'Died' else '⭐'
prediction_text = f"{emoji} {predicted_class}"


# Increase font size using Markdown and HTML
st.markdown(f"<h2 style='text-align: center; color: black;'>{prediction_text}</h2>", unsafe_allow_html=True)


# Display prediction probabilities with custom labels
st.subheader('Prediction Probability')
for label, prob in zip(prediction_labels, prediction_proba[0]):
    prob_percentage = int(prob * 100)  # Convert probability to percentage (integer)
    st.write(f"{label}: {prob_percentage}%")






