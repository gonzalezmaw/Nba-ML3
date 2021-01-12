import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from SLR import SLR
from RFR import RFR
from SVMC import SVMC
from NBC import NBC
from KNN import KNNC
from RFC import RFC
from PRM import PRM
from MRFR import MRFR
from MLPR import MLPR
from MLPC import MLPC
from KMeans import KMeans


st.title("Machine Learning for Everyone")

st.sidebar.write("""
# **Nubila ML**

""")
modelSelect_name = st.sidebar.selectbox(
    "Select a Model", ("Simple Linear Regression", "Polynomial Regression", "Random Forest Regression",
                       "Random Forest Classification", "Support Vector Machines", "Naive Bayes Classification",
                       "K-Nearest Neighbors", "NN - Multi-Layer Perceptron", "K-Means Clustering"))


if modelSelect_name == "Simple Linear Regression":
    st.write("""
        ## **Simple Linear Regression Model**

        """)
    st.write("""
    ### **Simple Regression Method**

    """)
    SLR()
elif modelSelect_name == "Random Forest Regression":
    st.write("""
        ## **Random Forest Regression Model**

        """)

    modelSelect_Type = st.sidebar.selectbox(
        "Select a Method", ("Single Variable Regression", "Multiple Variable Regression"))

    if modelSelect_Type == "Single Variable Regression":

        st.write("""
        ### **Simple Regression Method**

        """)
        RFR()
    else:
        st.write("""
        ### **Multiple Regression Method**

        """)
        MRFR()

elif modelSelect_name == "Support Vector Machines":
    st.write("""
        ## **Suport Vector Machines Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    SVMC()

elif modelSelect_name == "Naive Bayes Classification":
    st.write("""
        ## **Naive Bayes Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    NBC()

elif modelSelect_name == "K-Nearest Neighbors":
    st.write("""
        ## **K-Nearest Neighbors Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    KNNC()

elif modelSelect_name == "Random Forest Classification":
    st.write("""
        ## **Random Forest Classification Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    RFC()

elif modelSelect_name == "Polynomial Regression":
    st.write("""
        ## **Polynomial Regression Model**

        """)
    st.write("""
    ### **Single-Variable Regression Method**

    """)
    PRM()

elif modelSelect_name == "Random Forest Regression":
    st.write("""
        ## **Multiple Random Forest Regression Model**

        """)
    st.write("""
    ### **Multiple Variable Regression Method**

    """)
    MRFR()

elif modelSelect_name == "NN - Multi-Layer Perceptron":

    modelSelect_Type = st.sidebar.selectbox(
        "Select a Method", ("Neural Network Regression", "Neural Network Classification"))

    if modelSelect_Type == "Neural Network Regression":

        st.write("""
        ## **Multi-Layer Perceptron Regressor Model**

        """)

        st.write("""
    ### **Neural Network (supervised) Regression Method**

    """)
        MLPR()
    else:
        st.write("""
        ## **Multi-Layer Perceptron Classification Model**

        """)
        st.write("""
    ### **Neural Network (supervised) Classification Method**

    """)
        MLPC()

elif modelSelect_name == "K-Means Clustering":
    st.write("""
        ## **K-Means Clustering Model**

        """)

    st.write("""
    ### **COMING SOON**

    """)
    KMeans()
