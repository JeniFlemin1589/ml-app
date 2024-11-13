import streamlit as st 
from pycaret.classification import load_model

pipeline  = load_model("Trained_model")
pipeline