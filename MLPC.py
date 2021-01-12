import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from scipy.stats import pearsonr
from scipy import stats
from sklearn import preprocessing
from random import randint
import sklearn as sk
import sklearn.neural_network
import math
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


def MLPC():  # Multi-Layer Perceptron Classifier
    try:
        parameter_test_size = st.sidebar.slider(
            "Test size (fraction)", 0.02, 0.80, 0.2)
        st.sidebar.write("Test size: ", parameter_test_size)

        activation = st.sidebar.selectbox(
            "Select a activation function", ("identity", "logistic sigmoid", "hyperbolic tan", "rectified linear unit"))

        if activation == "identity":
            activation = "identity"
        elif activation == "logistic sigmoid":
            activation = "logistic"
        elif activation == "hyperbolic tan":
            activation == "tanh"
        elif activation == "rectified linear unit":
            activation = "relu"

        solver = st.sidebar.selectbox("Select a solver", ("quasi-Newton optimizer",
                                                          "stochastic gradient descent", "stochastic gradient(adam)"))

        if solver == "quasi-Newton optimizer":
            solver = "lbfgs"
        elif solver == "stochastic gradient descent":
            solver = "sgd"
        elif solver == "stochastic gradient(adam)":
            solver = "adam"

        learning_rate_init = st.sidebar.text_input(
            "Initial learning rate", 0.001)
        lri = float(learning_rate_init)

        hidden_layer_sizes = st.sidebar.text_input(
            "Hidden layer sizes", "5,5,5")

        max_iter = st.sidebar.number_input("Maximum number of iterations", 200)
        tol = st.sidebar.text_input("Tolerance for the optimization", 1e-4)
        tol = float(tol)

        nn = list(eval(hidden_layer_sizes))

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

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=parameter_test_size, random_state=1234)
            st.write("Complete data: ", len(df))
            st.write("Data to train: ", len(X_train))
            st.write("Data to test: ", len(X_test))
            st.write(df.describe())

            if showData:
                st.subheader('Dataset')
                st.write(df)

            # model = MLPRegressor(hidden_layer_sizes=100,
                # learning_rate_init = 0.001, max_iter = 200)
                # hidden_layer_sizes=nn[:]
            model = sk.neural_network.MLPClassifier(hidden_layer_sizes=nn[:],
                                                    activation=activation,
                                                    solver=solver,
                                                    alpha=0.0001,
                                                    batch_size='auto',
                                                    learning_rate='constant',
                                                    learning_rate_init=lri,
                                                    power_t=0.5,
                                                    max_iter=max_iter,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    tol=tol,
                                                    verbose=False,
                                                    warm_start=False,
                                                    momentum=0.9,
                                                    nesterovs_momentum=True,
                                                    early_stopping=False,
                                                    validation_fraction=0.1,
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    epsilon=1e-08,
                                                    n_iter_no_change=10,
                                                    max_fun=15000)

            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = model.predict(X_test)

            st.subheader("""
                            **Classification**
                            """)
            correlation = round(pearsonr(y_pred, y_test)[0], 5)
            st.write("Pearson correlation coefficient:")
            st.info(correlation)

            # plot data

            acc = round(accuracy_score(y_test, y_pred), 5)

            # PLOT
            pca = PCA(2)
            X_projected = pca.fit_transform(X)

            x1 = X_projected[:, 0]
            x2 = X_projected[:, 1]

            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure()
            # fig, ax = plt.subplots()
            # figure, ax = plt.subplots()
            plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
            plt.xlabel("Principal Component X1")
            plt.ylabel("Principal Component X2")
            plt.title('K-Nearest Neighbors Classification')
            plt.colorbar()
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.info(classification_report(y_test, y_pred))

            st.write("**Accuracy:**")
            st.info(acc)

            showBloxplot = st.checkbox('Show Boxplot')
            numFiles = math.ceil(df.shape[1]/5)

            if showBloxplot:
                fig, axs = plt.subplots(
                    ncols=5, nrows=numFiles, figsize=(20, 10))
                index = 0
                axs = axs.flatten()
                for k, v in df.items():
                    # sns.boxplot(y=k, data=df, ax=axs[index], palette="Paired")
                    sns.boxplot(y=k, data=df, ax=axs[index], palette="Set3")
                    index += 1
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
                st.pyplot(fig)

            showOutPerc = st.checkbox('Show Outliers Percentage')
            if showOutPerc:
                for k, v in df.items():
                    q1 = v.quantile(0.25)
                    q3 = v.quantile(0.75)
                    irq = q3 - q1
                    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
                    perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
                    st.info("Column %s outliers = %.2f%%" % (k, perc))

                    # Columns like CRIM, ZN, RM, B seems to have outliers. Let's see the outliers percentage in every column.

            ymax = np.amax(y)

            # Let's remove MEDV outliers (MEDV = 50.0) before plotting more distributions
            df = df[~(df[y.name] >= ymax)]

            numFilesX = math.ceil(X.shape[1]/5)

            showHistograms = st.checkbox('Show Histograms')
            if showHistograms:
                fig2, axs = plt.subplots(
                    ncols=5, nrows=numFilesX, figsize=(20, 10))
                index = 0
                axs = axs.flatten()
                for k, v in df.items():
                    sns.distplot(v, ax=axs[index], color="dodgerblue")
                    index += 1
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
                st.pyplot(fig2)

                # fig, ax = plt.subplots()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            showMapHeat = st.checkbox('Show Matrix Correlations')

            if showMapHeat:
                plt.figure(figsize=(20, 10))
                sns.heatmap(df.corr().abs(),  annot=True, cmap="Blues")
                st.pyplot()

            showCorrChart = st.checkbox('Show Correlation Charts')

            if showCorrChart:
                # Let's scale the columns before plotting them against MEDV
                min_max_scaler = preprocessing.MinMaxScaler()
                column_sels = list(X.columns)
                # st.info(column_sels)
                # x = df.loc[:, column_sels]
                # y = df[y.name]
                X = pd.DataFrame(data=min_max_scaler.fit_transform(X),

                                 columns=column_sels)
                fig, axs = plt.subplots(
                    ncols=5, nrows=numFiles, figsize=(20, 10))
                index = 0
                axs = axs.flatten()
                colors = "bgrcmyk"
                # color_index = 0
                for i, k in enumerate(column_sels):
                    # sns.regplot(y=y, x=X[k], ax=axs[i], color="blue")
                    sns.regplot(y=y, x=X[k], ax=axs[i],
                                color=colors[randint(0, 6)])
                    # color_index += 1
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
                st.pyplot(fig)

            st.subheader("Prediction:")
            uploaded_file_target = st.file_uploader(
                "Choose a new file for prediction")
            if uploaded_file_target is not None:
                dfT = pd.read_csv(uploaded_file_target)

                # Using all column except for the last column as X
                X_new = dfT.iloc[:, :-1]
                y_new = model.predict(X_new)
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
                plt.title('KNN - Prediction')
                plt.colorbar()
                st.pyplot()

        else:
            st.info('Awaiting for CSV file to be uploaded.')

    except:
        st.info('ERROR - Please check your Dataset, parameters o selected model')
    return
