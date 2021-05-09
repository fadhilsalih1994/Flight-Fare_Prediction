
"""
__author__ = 'Fadhil Salih'
__email__ = 'fadhilsalih94@gmail.com'
__date__ = '2021-05-08'
__dataset__ = 'https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh/'
__connect__ = 'https://www.linkedin.com/in/fadhilsalih/'
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle


def date_preprocessing(df, col):
    """Function to convert date to date time objects (Day and Month only since the data is from the same year)
    :param df: input dataframe
    :param col: input variable
    :return: df
    """
    df["Journey_Day"] = pd.to_datetime(df[col], format="%d/%m/%Y").dt.day
    df["Journey_Month"] = pd.to_datetime(df[col], format="%d/%m/%Y").dt.month
    df.drop([col], axis=1, inplace=True)
    return(df)


def time_preprocessing(df, col):
    """Function to convert departure & arrival time to date time objects
    :param df: input dataframe
    :param col: input variable
    :return: df
    """
    df[col.split(sep="_")[0] + "_hour"] = pd.to_datetime(df[col]).dt.hour

    # Extracting Minutes
    df[col.split(sep="_")[0] + "_min"] = pd.to_datetime(df[col]).dt.minute

    # Now we can drop Dep_Time as it is of no use
    df.drop([col], axis=1, inplace=True)
    return(df)


def duration_preprocessing(df):
    """Function to extract hours and minutes from the flight duration in order to create separate datetime variables for them
    :param df: input dataframe
    :return: None
    """

    duration = list(df["Duration"])

    for i in range(len(duration)):
        if len(duration[i].split()) != 2:
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"
            else:
                duration[i] = "0h " + duration[i]

    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep="h")[0]))
        duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))

    df["Duration_hours"] = duration_hours
    df["Duration_mins"] = duration_mins

    df.drop(["Duration"], axis=1, inplace=True)

    return(df)


def heat_map_fit(df):
    """Function to create a heatmap based on the correlations with the right dimensions
    :param df: input dataframe
    :return: None
    """

    # Set required font size
    fontsize_pt = 2
    dpi = 72.27

    # comput the matrix height in points and inches
    matrix_height_pt = fontsize_pt * float(df.shape[0])
    matrix_height_in = matrix_height_pt / dpi

    # compute the required figure height
    top_margin = 0.04  # in percentage of the figure height
    bottom_margin = 0.04  # in percentage of the figure height
    figure_height = matrix_height_in / (1 - top_margin - bottom_margin)

    # build the figure instance with the desired height
    fig, ax = plt.subplots(
        figsize=(15, figure_height),
        gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))

    heat = sns.heatmap(df.corr(), ax=ax, annot=True, cmap="RdYlGn")
    heat.set_yticklabels(heat.get_yticklabels(), rotation=0, fontsize=8)
    heat.set_xticklabels(heat.get_xticklabels(), rotation=0, fontsize=8)
    plt.show()


def exploratory_data_analysis(df):
    """Function to perform some exploratory data analysis on the dataset
    :param df: input dataframe
    :return: None
    """

    # Airline vs Price
    sns.catplot(y="Price", x="Airline", data=df.sort_values(
        "Price", ascending=False), kind="boxen", height=6, aspect=3)
    plt.show()

    # Airline vs Price
    sns.catplot(y="Price", x="Source", data=df.sort_values(
        "Price", ascending=False), kind="boxen", height=6, aspect=3)
    plt.show()

    # Airline vs Price
    sns.catplot(y="Price", x="Destination", data=df.sort_values(
        "Price", ascending=False), kind="boxen", height=6, aspect=3)
    plt.show()

    heat_map_fit(df)


def one_hot_encode(df, col):
    """Function to one-hot encode categorical variables
    :param df: input dataframe
    :return: df
    """

    df_final = pd.get_dummies(data=df, columns=col, drop_first=True)

    return(df_final)


def preprocessing_main(df):
    """Function to preprocess the dataset in order to train the model
    :param df: input dataframe
    :return: df_final
    """

    df.dropna(inplace=True)

    df = date_preprocessing(
        df, "Date_of_Journey")

    df = time_preprocessing(
        df, "Dep_Time")

    df = time_preprocessing(
        df, "Arrival_Time")

    df = duration_preprocessing(df)

    df.replace(
        {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)
    df.drop(
        ["Route", "Additional_Info"], axis=1, inplace=True)

    if 'Price' in df.columns:
        exploratory_data_analysis(df)
    else:
        pass

    encode_list = ['Airline', 'Source', 'Destination']

    df_final = one_hot_encode(df, encode_list)

    return(df_final)


def target_variable_split(df):
    """Function to split dataset into predictor variables and target variable
    :param df: input dataframe
    :return: X, y
    """
    X = df.loc[:, df.columns != 'Price']
    y = df.iloc[:, df.columns == 'Price']
    return (X, y)


def important_features(df):
    """Function to plot graph of important features for better visualization
    :param df: input dataframe
    :return: None
    """
    X, y = target_variable_split(df)
    model = ExtraTreesRegressor()
    model.fit(X, y.values.ravel())
    plt.figure(figsize=(12, 8))
    important = pd.Series(model.feature_importances_, index=X.columns)
    important.nlargest(20).plot(kind='barh')
    plt.show()


def test_train_split(df):
    """Function to split the train and test dataset
    :param df: input dataframe
    :return: None
    """
    X, y = target_variable_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    return(X_train, X_test, y_train, y_test)


def hyperparameter_selection_rf():
    """Function to select hyperparameters for random forest regressor model
    :param: None
    :return: random_grid
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    return(random_grid)


def final_model(df):
    """Function to train the model and call the save_pickle function
    :param: rf_random, X_test, y_test
    :return: None
    """

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=hyperparameter_selection_rf(),
                                   scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
    X_train, X_test, y_train, y_test = test_train_split(df)
    rf_random.fit(X_train, y_train.values.ravel())

    results(rf_random, X_train, y_train, X_test, y_test)
    save_pickle(rf_random)


def results(rf_random, X_train, y_train, X_test, y_test):
    """Function to produce results of the trained model
    :param: rf_random, X_train, y_train,X_test, y_test
    :return: None
    """

    print(rf_random.best_params_)
    predictions = rf_random.predict(X_test)
    print(
        f"R Squared Score of Test Set: {round(metrics.r2_score(y_test, predictions)*100,2)}%")
    sns.displot(y_test.values.ravel()-predictions, kde=True)
    plt.show()

    plt.scatter(y_test.values.ravel(), predictions, alpha=0.5)
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.title("Scatterplot")
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(
        y_test.values.ravel(), predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(
        y_test.values.ravel(), predictions))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test.values.ravel(), predictions)))


def save_pickle(rf_random):
    """Function to pickle the model
    :param: rf_random
    :return: None
    """
    # open a file to store the model
    file = open('random_forest_regression_model.pkl', 'wb')
    # dump information to that file
    pickle.dump(rf_random, file)


def load_pickle(df):
    """Function to load pickled model
    :param: None
    :return: None
    """
    # Open the file containing the trained model
    file = open('random_forest_regression_model.pkl', 'rb')
    # Load model
    model = pickle.load(file)

    X_train, X_test, y_train, y_test = test_train_split(df)

    prediction = model.predict(X_test)

    print(
        f"R Squared Score of Test after loading pickle: {round(metrics.r2_score(y_test, prediction)*100,2)}%")


def main():

    flight_train = pd.read_excel(r"Data_Train.xlsx")
    flight_test = pd.read_excel(r"Test_set.xlsx")

    flight_train_processed = preprocessing_main(flight_train)

    flight_test_processed = preprocessing_main(flight_test)

    important_features(flight_train_processed)

    final_model(flight_train_processed)

    load_pickle(flight_train_processed)

    # load_pickle(flight_test_processed)


if __name__ == '__main__':
    main()
