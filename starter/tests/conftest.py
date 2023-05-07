import pytest
import pandas as pd
import pickle
from ml_model.ml.data import process_data
from sklearn.model_selection import train_test_split


@pytest.fixture(scope='session')
def data():

    data_path = './ml_model/data/census.csv'
    df = pd.read_csv(data_path)

    return df

@pytest.fixture(scope="session")
def cat_features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = [    "workclass",
                        "education",
                        "marital-status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native-country"]
    return cat_features

@pytest.fixture(scope="session")
def features():
    """
    Fixture - will return the categorical features as argument
    """
    features = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'salary']
    return features

@pytest.fixture(scope='session')
def model():

    model_path = './ml_model/model/model.pkl'
    model = pickle.load(open(model_path), 'rb')

    return model


@pytest.fixture(scope="session")
def dataset_split(data, cat_features):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=cat_features,
                                            label="salary",
                                            training=True
                                        )
    X_test, y_test, encoder, lb = process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)