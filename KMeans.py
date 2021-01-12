import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster


def KMeans():
    try:

        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            X = df.iloc[:, :]
            #X = X.astype(str)

            st.write(X)

            st.write(df.describe())

        # penguins = sns.load_dataset(pd)

            # fig = sns.pairplot(df)
            fig = sns.pairplot(df)

        # fig = sns.pairplot(df[['Age', 'Spending Score']])
            st.pyplot(fig)

            # kmeans = cluster.KMeans(n_clusters=5, init="k-means++")
            # kmeans = kmeans.fit(df)
            # st.write(kmeans.cluster_centers_)
            st.write("""
        # **COMING SOON**

        """)

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')

    return

    # cargar el conjunto de datos
