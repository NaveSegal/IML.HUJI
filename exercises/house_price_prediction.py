import numpy
import sklearn.linear_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# from sklearn.linear_model import LinearRegression

pio.templates.default = "simple_white"
CAT_SORT_BY_CORR = ['bedrooms_11', 'bathrooms_7.5', 'bedrooms_33', 'bathrooms_1.25', 'grade_8.0', 'bathrooms_2.25',
                    'long', 'grade_1.0', 'condition_3.0', 'bedrooms_10', 'condition_0.0', 'grade_0.0', 'bathrooms_0.0',
                    'grade_3.0', 'bathrooms_0.5', 'bedrooms_0', 'floors_0.0_left', 'floors_0.0_right',
                    'floors_1.5_left', 'floors_1.5_right', 'bedrooms_9', 'floors_3.0_left', 'floors_3.0_right',
                    'floors_3.5_left', 'floors_3.5_right', 'condition_1.0', 'bathrooms_2.5', 'bathrooms_6.5',
                    'condition_4.0', 'grade_4.0', 'bedrooms_8', 'bathrooms_0.75', 'bedrooms_7', 'condition_2.0',
                    'yr_built', 'bathrooms_6.75', 'condition_5.0', 'bedrooms_1', 'bathrooms_6.25', 'bathrooms_2.0',
                    'bathrooms_5.75', 'bathrooms_2.75', 'sqft_lot15', 'grade_5.0', 'bathrooms_5.25', 'bathrooms_3.0',
                    'bedrooms_6', 'sqft_lot', 'bathrooms_1.75', 'bathrooms_1.5', 'bathrooms_5.0', 'bathrooms_6.0',
                    'bathrooms_5.5', 'bathrooms_8.0', 'bathrooms_7.75', 'floors_2.5_left', 'floors_2.5_right',
                    'yr_renovated', 'lat', 'bathrooms_4.75', 'bedrooms_2', 'bathrooms_4.5', 'bathrooms_3.75',
                    'bathrooms_4.0', 'bathrooms_4.25', 'bedrooms_4', 'bedrooms_3', 'bedrooms_5', 'bathrooms_3.25',
                    'bathrooms_3.5', 'grade_6.0', 'grade_13.0', 'floors_2.0_left', 'floors_2.0_right', 'grade_9.0',
                    'bathrooms_1.0', 'floors_1.0_left', 'floors_1.0_right', 'waterfront', 'grade_12.0', 'grade_7.0',
                    'sqft_basement', 'grade_10.0', 'grade_11.0', 'view', 'sqft_living15', 'sqft_above', 'sqft_living']


def get_days_between_dates(dates1: pd.Series, dates2: pd.Series):
    return (dates1 - dates2).apply(lambda period: period.days)


def get_years_between_dates(dates1: pd.Series, dates2: pd.Series):
    return (dates1 - dates2).apply(lambda period: period.years)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    TOO_BE_DROPED = ["id", "date", ]
    full_data = pd.read_csv(filename).drop_duplicates()  # .astype({"date": "datetime64"})  # todo ?? add .reshape(-1,1)
    features = clean_data(full_data)
    grade, condition = features["grade"], features["condition"],

    features = pre_process(features)
    features["grade"] = grade
    features["condition"] = condition

    labels = features["price"]
    return features.drop(["price"] + ["id"], axis='columns'), labels


def clean_data(data: pd.DataFrame):
    data.drop(data[data.price > 1000].index)
    data.drop(data[data.bedrooms >= 0].index)
    data.drop(data[data.bathrooms >= 0].index)
    data.drop(data[data.sqft_living > 100].index)
    data.drop(data[data.sqft_lot15 > 0].index)
    data.drop(data[data.yr_built > 1650].index)
    data.drop(data[data.yr_built < 2022].index)
    data.date.dropna()
    return data


def cat_onehot_encoding(X: pd.DataFrame, cats: list):
    for cat in cats:
        encoder_df = pd.get_dummies(X[cat], prefix=f"{cat}")
        X = X.join(encoder_df, lsuffix='_left', rsuffix='_right')

    return X.drop(cats, axis='columns')


def pre_process(data: pd.DataFrame):
    # todo pre process the data. one hot endoding and etc.
    # data["date"] = data["date"].apply(lambda x: int(float(str(x)[:4])))
    # data["years_since_built"] = get_years_between_dates(data["date"], data["yr_built"])
    # data["years_since_renovation"] = get_years_between_dates(data["date"], data["ye_renovated"])
    # data["years_since_renovation"] = data["years_since_renovation"].fillna(0)  # todo check if they are nan or else neg
    print(data.columns)
    data.drop(["yr_built", "lat", "long"], axis='columns')  # todo clust long and lat into buckets and not dell them...
    CATEGORICAL = ["floors", "bathrooms", "bedrooms", "floors", "condition", "grade", ]
    data = cat_onehot_encoding(data, CATEGORICAL)
    return data.drop(["zipcode", "date"] + CAT_SORT_BY_CORR[:int((len(CAT_SORT_BY_CORR) / 4))], axis="columns")


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # todo choose two features and add them to the list beast one best one worst
    feature_list = list(X)
    sig_y = y.std()
    corr_dict = dict()

    for feature in feature_list:
        cov = X[feature].cov(y)
        std_multi = (sig_y * X[feature].std())
        corr_dict[feature] = (cov / std_multi)

    # print({k: v for k, v in sorted(corr_dict.items(), key=lambda item: abs(item[1]))}.keys())
    plt.bar(*zip(*corr_dict.items()))
    plt.title(f'feature Pearson correlation of {feature} with prices')
    plt.xlabel("features")
    plt.ylabel("Pearson correlation values")
    plt.savefig(output_path)
    plt.show()
    print("Correlation info:\n", corr_dict)


def evaluation_stage(X, y, output_path="."):
    dict_of_result = {}

    for fraction in range(10, 101):  # todo change to 1,
        dict_of_result[fraction] = []
        data = X.sample(frac=(1 / fraction))
        label = y.sample(frac=(1 / fraction))

        for _ in range(10):
            model = LinearRegression()
            model.fit(data, label)
            dict_of_result[fraction].append(model._loss(data, label))

    print("here the items are: ", [(k, v) for k, v in dict_of_result.items()])  # TODO Debug this whole section
    avg_var_dict = {key: (sum(value) / len(value), numpy.var(value)) for key, value in dict_of_result.items()}

    finalito_avg = {key: value[0] for key, value in avg_var_dict.items()}
    print([(k, v) for k, v in finalito_avg.items()])
    plot_this = {key: (value[0] - 2 * value[1], value[0] + 2 * value[1]) for key, value in avg_var_dict.items()}
    print("here comes the ribbon", plot_this.values())

    # plt.fill_between(plot_this.keys(), finalito_avg.items())
    # plt.plot(plot_this.keys(), finalito_avg.values())
    # # where=[finalito_avg.values() > val[0] & finalito_avg.values() < val[1] for val in plot_this.values()]
    # plt.show()
    # plt.bar(*zip(*plot_this.items()))
    # plt.title(f'The Mean Loss Function by Fraction of the data')
    # plt.xlabel("Fraction of the data")
    # plt.ylabel("Loss Function")
    # plt.savefig(output_path)
    # plt.show()

    fig = go.Figure([go.Scatter(x=X, y=y, name="Real Model", showlegend=True,
                                marker=dict(color="black", opacity=.7),
                                line=dict(color="black", dash="dash", width=1))],
                    layout=go.Layout(title=r"$\text{(1) Simulated Data}$",
                                     xaxis={"title": "x - Explanatory Variable"},
                                     yaxis={"title": "y - Response"},
                                     height=400))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    _PATH_ = r"C:\Users\user\PycharmProjects\IML.HUJI\datasets\house_prices.csv"
    # Question 1 - Load and preprocessing of housing prices dataset
    print("\nLoad and preprocessing of housing prices dataset\n")
    df, price_labels = load_data(_PATH_)

    # Question 2 - Feature evaluation with respect to response
    print("\nFeature evaluation with respect to response\n")
    feature_evaluation(df, price_labels)
    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, price_labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    evaluation_stage(train_X, train_y)

# drop them: ['bedrooms_11', 'bathrooms_7.5', 'bedrooms_33', 'bathrooms_1.25', 'grade_8.0', 'bathrooms_2.25', 'long', 'grade_1.0', 'condition_3.0', 'bedrooms_10', 'condition_0.0', 'grade_0.0', 'bathrooms_0.0', 'grade_3.0', 'bathrooms_0.5', 'bedrooms_0', 'floors_0.0_left', 'floors_0.0_right', 'floors_1.5_left', 'floors_1.5_right', 'bedrooms_9', 'floors_3.0_left', 'floors_3.0_right', 'floors_3.5_left', 'floors_3.5_right', 'condition_1.0', 'bathrooms_2.5']
CAT_SORT_BY_CORR = ['bedrooms_11', 'bathrooms_7.5', 'bedrooms_33', 'bathrooms_1.25', 'grade_8.0', 'bathrooms_2.25',
                    'long', 'grade_1.0', 'condition_3.0', 'bedrooms_10', 'condition_0.0', 'grade_0.0', 'bathrooms_0.0',
                    'grade_3.0', 'bathrooms_0.5', 'bedrooms_0', 'floors_0.0_left', 'floors_0.0_right',
                    'floors_1.5_left', 'floors_1.5_right', 'bedrooms_9', 'floors_3.0_left', 'floors_3.0_right',
                    'floors_3.5_left', 'floors_3.5_right', 'condition_1.0', 'bathrooms_2.5', 'bathrooms_6.5',
                    'condition_4.0', 'grade_4.0', 'bedrooms_8', 'bathrooms_0.75', 'bedrooms_7', 'condition_2.0',
                    'yr_built', 'bathrooms_6.75', 'condition_5.0', 'bedrooms_1', 'bathrooms_6.25', 'bathrooms_2.0',
                    'bathrooms_5.75', 'bathrooms_2.75', 'sqft_lot15', 'grade_5.0', 'bathrooms_5.25', 'bathrooms_3.0',
                    'bedrooms_6', 'sqft_lot', 'bathrooms_1.75', 'bathrooms_1.5', 'bathrooms_5.0', 'bathrooms_6.0',
                    'bathrooms_5.5', 'bathrooms_8.0', 'bathrooms_7.75', 'floors_2.5_left', 'floors_2.5_right',
                    'yr_renovated', 'lat', 'bathrooms_4.75', 'bedrooms_2', 'bathrooms_4.5', 'bathrooms_3.75',
                    'bathrooms_4.0', 'bathrooms_4.25', 'bedrooms_4', 'bedrooms_3', 'bedrooms_5', 'bathrooms_3.25',
                    'bathrooms_3.5', 'grade_6.0', 'grade_13.0', 'floors_2.0_left', 'floors_2.0_right', 'grade_9.0',
                    'bathrooms_1.0', 'floors_1.0_left', 'floors_1.0_right', 'waterfront', 'grade_12.0', 'grade_7.0',
                    'sqft_basement', 'grade_10.0', 'grade_11.0', 'view', 'sqft_living15', 'sqft_above', 'sqft_living']
