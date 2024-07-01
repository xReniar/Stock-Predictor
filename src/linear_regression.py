from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from data import download_stock_data, update_db_stock, pd, np
import joblib


def preprocessing(data: pd.Series, sequence_length:int) -> list:
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length].values
        target = data.iloc[i+sequence_length]
        sequences.append((sequence, target))
    return sequences

def split_data(sequences):
    return train_test_split(sequences,
                            test_size=0.2,
                            shuffle=False)

def create_training_data(train_data, test_data):
    # Prepare training data
    X_train = np.array([item[0] for item in train_data])
    y_train = np.array([item[1] for item in train_data])

    # Prepare testing data
    X_test = np.array([item[0] for item in test_data])
    y_test = np.array([item[1] for item in test_data])

    return (X_train, y_train, X_test, y_test)

def train(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(ticker:str, model:LinearRegression,X_test, y_test):
    last_sequence = X_test[-1].reshape(1, -1)
    predicted_price = model.predict(last_sequence)[0]
    mse = mean_squared_error(y_test,model.predict(X_test))

    update_db_stock(ticker,"linear_regression",predicted_price,mse)

def main(ticker):
    lookback = 10
    data = download_stock_data(ticker)
    sequences = preprocessing(data,lookback)
    train_data, test_data = split_data(sequences)

    X_train, y_train, X_test, y_test = create_training_data(train_data,test_data)
    model = train(X_train, y_train)
    evaluate_model(ticker,model,X_test, y_test)

    # saving model
    joblib.dump(model,f"../models/{ticker}.pkl")