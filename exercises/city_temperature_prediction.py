import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from IMLearn.metrics import loss_functions as ls

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    cols = ['Country', 'City', 'Date', 'Year', 'Month', 'Day', 'Temp']
    full_data = pd.read_csv(filename).drop_duplicates().astype({'Date': 'datetime64'})
    labels = full_data["Temp"]
    clean_df = clean_data(full_data)
    preproccess_df = preprocess_data(clean_df)

    return preproccess_df.drop(["Temp"], axis='columns'), labels


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(df[df.Year < 1900].index)
    df = df.drop(df[df.Month < 1].index)
    df = df.drop(df[df.Month > 12].index)
    df = df.drop(df[df.Day > 31].index)
    df = df.drop(df[df.Day < 1].index)
    df = df.drop(df[df.Temp < -40].index)
    df = df.drop(df[df.Temp > 50].index)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["DayOfYear"] = df["Date"].dt.dayofyear  # todo is this the right syntax?
    return df


def q2(X: pd.DataFrame, y):
    df = X.drop(X[X.Country != "Israel"].index)
    labels = y.drop(X[X.Country != "Israel"].index)
    # model = PolynomialFitting(k=3)  # 3 because this is how it looks like
    # model.fit(df, labels)
    df["labels"] = labels
    df["Year"] = df["Year"].astype(str)
    fig = px.scatter(df, x="DayOfYear", y="labels", color="Year", title="Temp by the day on year, colored by year")
    fig.show()

    std_val = df.groupby(by="Month").std()
    fig2 = px.scatter(std_val, x=list(range(1, 13)), y="labels", title="STD of monthly Temp over Moth of Year")
    fig2.show()


def q3(df: pd.DataFrame, y):
    df["Temp"] = y
    mean_grouped = df.groupby(by=["Country", "Month"]).mean().reset_index()
    std_grouped = df.groupby(by=["Country", "Month"]).std().reset_index()
    mean_grouped["std_err"] = std_grouped["Temp"]

    fig2 = plotly.express.line(mean_grouped, x=list(range(1, 49)), y=["Country", "Month", "Temp"], error_y="std_err",
                               color="Country", title="Mean of monthly Temp by Country")
    fig2.show()


def q4(df, labels):
    df["Temp"] = labels
    df = df.drop(df[df.Country != "Israel"].index)
    train_X, train_y, test_X, test_y = split_train_test(df, df["Temp"])
    print("HELOOOO\n")
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)

    poly_err_deg = {k: 0 for k in range(1, 11)}

    for k in poly_err_deg.keys():
        model = PolynomialFitting(k)
        model.fit(train_X["DayOfYear"], train_y)
        poly_err_deg[k] = model._loss(test_X["DayOfYear"], test_y)

    fig = px.bar(x=poly_err_deg.keys(), y=poly_err_deg.values())
    fig.show()
    print("this is the error:", poly_err_deg)


def get_israeli_model(df, labels):
    df["Temp"] = labels
    df = df.drop(df[df.Country != "Israel"].index)
    train_X, train_y, test_X, test_y = split_train_test(df, df["Temp"])
    model = PolynomialFitting(3)
    model.fit(train_X["DayOfYear"], train_y)
    return model


def q5(df, labels):
    df["Temp"] = labels

    poly_err_deg = {country: 0 for country in df.Country}
    model = get_israeli_model(df, labels)
    for country in poly_err_deg.keys():
        cur_df = df.drop(df[df.Country != f"{country}"].index)
        train_X, train_y, test_X, test_y = split_train_test(cur_df, cur_df["Temp"])
        poly_err_deg[country] = model._loss(test_X["DayOfYear"], test_y)

    fig = px.bar(x=poly_err_deg.keys(), y=poly_err_deg.values())
    fig.show()
    print("this is the error:", poly_err_deg)


def _mse_(y_hat, y):
    res = np.sum(((y_hat - y) ** 2)) / len(y)
    print(res)
    return res


if __name__ == '__main__':
    np.random.seed(0)
    file_name = r"C:\Users\user\PycharmProjects\IML.HUJI\datasets\City_Temperature.csv"
    # Question 1 - Load and preprocessing of city temperature dataset
    df, labels = load_data(file_name)

    # Question 2 - Exploring data for specific country
    q2(df, labels)

    # Question 3 - Exploring differences between countries
    q3(df, labels)

    # Question 4 - Fitting model for different values of `k`
    q4(df, labels)
    # Question 5 - Evaluating fitted model on different countries
    q5(df, labels)
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    _mse_(y_true, y_pred)

