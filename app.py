import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from sklearn.preprocessing import LabelEncoder

# Sidebar for navigation
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, and PyCaret. And It's Great!")

# Check if sourcedata.csv exists
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Upload data
if choice == "Upload":
    st.title("Upload your data for modeling!")
    file = st.file_uploader("Upload your file here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

# Profiling
if choice == "Profiling":
    if 'df' in locals():
        st.title("Automated Exploratory Data Analysis")
        
        # Sample the dataset if it's too large
        if df.shape[0] > 10000:  # You can adjust the sample size threshold as needed
            df_sample = df.sample(n=10000, random_state=42)
            st.warning("Dataset too large, sampling 10,000 rows for profiling.")
        else:
            df_sample = df
        
        # Adjust profile report settings
        profile_report = ProfileReport(
            df_sample,
            minimal=True  # Set to True for less detailed and faster profiling
        )
        
        st_profile_report(profile_report)
    else:
        st.error("Please upload a dataset first.")

if choice == "ML": 
    st.title("Machine Learning Go Magic!")
    target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Train Model'): 
        df_cleaned = df.dropna(subset=[target])
        
        # Filter out classes with less than 2 instances
        class_counts = df_cleaned[target].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        df_filtered = df_cleaned[df_cleaned[target].isin(valid_classes)]

        if len(df_filtered) < 2:
            st.error("Not enough data to run the model after filtering out classes with less than 2 instances.")
        else:
            # Encode categorical features
            df_encoded = df_filtered.copy()
            for col in df_encoded.select_dtypes(include='object').columns:
                if col != target:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
            
            setup(df_encoded, target=target)

            set_up = pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(set_up)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            st.write(best_model)
            best_model
            save_model(best_model,"best_model")




if choice == "Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("Download the file", f, "Trained_model.pkl")
    pass
