import models

def train(model: models.LinearRegression | models.LSTM, X_train, y_train) -> models.LinearRegression | models.LSTM:
    if type(model) == models.LSTM:
        print("LSTM")

    if type(model) == models.LinearRegression:
        model.fit(X_train,y_train)
        return model