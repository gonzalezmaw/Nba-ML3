import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


def SVMC():
    # Support Vector Machine Classifier
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        param_C = st.sidebar.slider("C parameter", 0.01, 10.0)
        st.sidebar.write("C parameter: ", param_C)

        kernelType = st.sidebar.selectbox("Kernel type",
                                          ("Linear", "Polynomial", "Gaussian", "Sigmoid"))

        if kernelType == "Polynomial":
            degreePoly = st.sidebar.slider("Polynomial degree", 1, 10, 2)
            st.sidebar.write("Polynomial degree: ", degreePoly)

        st.sidebar.info("""
        [More information](http://gonzalezmaw.pythonanywhere.com/)
        """)

        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Using all column except for the last column as X
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]  # Selecting the last column as Y

            showData = st.checkbox('Show Dataset')

            st.write("shape of dataset:", df.shape)
            st.write('number of classes:', len(np.unique(y)))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=parameter_test_size)
            st.write("Complete data: ", len(df))
            st.write("Data to train: ", len(X_train))
            st.write("Data to test: ", len(X_test))

            if showData:
                st.subheader('Dataset')
                st.write(df)

            if kernelType == "Linear":
                kernelType = "linear"
            elif kernelType == "Polynomial":
                kernelType = "poly"
            elif kernelType == "Gaussian":
                kernelType = "rbf"
            elif kernelType == "Sigmoid":
                kernelType = "sigmoid"

            if kernelType == "Polynomial":
                svclassifier = SVC(
                    C=param_C, kernel=kernelType, degree=degreePoly)
            else:
                svclassifier = SVC(C=param_C, kernel=kernelType)

            svclassifier.fit(X_train, y_train)
            y_pred = svclassifier.predict(X_test)
            acc = round(accuracy_score(y_test, y_pred), 5)
            # precision_ = round(precision_score(y_test, y_pred), 5)

            # PLOT
            pca = PCA(2)
            X_projected = pca.fit_transform(X)

            x1 = X_projected[:, 0]
            x2 = X_projected[:, 1]

            st.subheader("Classification:")

            """figure, ax = plt.subplots()
            ax.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
            plt.xlabel("Principal Component X1")
            plt.ylabel("Principal Component X2")
            # plt.colorbar()
            plt.colorbar()
            st.pyplot(figure)
            st.set_option('deprecation.showPyplotGlobalUse', False)"""
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure()
            # fig, ax = plt.subplots()
            # figure, ax = plt.subplots()
            plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
            plt.xlabel("Principal Component X1")
            plt.ylabel("Principal Component X2")
            plt.title('Support Vector Machines Classifier')
            plt.colorbar()
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.info(classification_report(y_test, y_pred))

            st.write("**Accuracy:**")
            st.info(acc)
            # st.write("**Precision:**")
            uploaded_file_target = st.file_uploader(
                "Choose a new file for prediction")
            if uploaded_file_target is not None:
                dfT = pd.read_csv(uploaded_file_target)
                # st.write(dfT)
                # Using all column except for the last column as X
                X_new = dfT.iloc[:, :-1]
                y_new = svclassifier.predict(X_new)
                y_target = dfT.iloc[:, -1].name
                dfT_new = pd.DataFrame(X_new)
                dfT_new[y_target] = y_new  # Agregar la columna

                st.subheader("""
                        **Prediction**
                        """)

                st.write(dfT_new)

                pca2 = PCA(2)
                X_projected = pca2.fit_transform(X_new)

                x1_new = X_projected[:, 0]
                x2_new = X_projected[:, 1]

                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.figure()
                plt.scatter(x1_new, x2_new, c=y_new, alpha=0.8, cmap="viridis")
                plt.xlabel("Principal Component X1")
                plt.ylabel("Principal Component X2")
                plt.title('SVM - Prediction')
                plt.colorbar()
                st.pyplot()

                # st.info(classification_report(y_test, y_pred))

            # st.info(precision_)

        else:
            st.info('Awaiting for CSV file to be uploaded.')
    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
        return
