from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from data import download_stock_data, update_result_stock, pd, np
from torch import nn
import torch
import indicators


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_stacked_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
def preprocessing(data: pd.DataFrame, lookback: int):
    data = data.reset_index()
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    data["Date"] = data["Datetime"].dt.date
    data["Time"] = data["Datetime"].dt.hour
    data = data.drop(columns=["Volume","Adj Close","Datetime"])
    
    # adding indicator values
    indicators.sma(data,lookback)
    indicators.std(data,lookback)
    indicators.bb(data,lookback)
    indicators.ema(data,lookback)
    indicators.rsi(data,lookback)

    # removing unnecessary features
    features = data.columns.tolist()
    features.remove("Open")
    features.remove("Close")
    features.append("Close")
    features.remove("EMA-20")

    # selecting features
    data = data.dropna()
    data = data[features]
    data = data.set_index("Date")

    # shifting values
    data["Close"] = data["Close"].shift(-1)
    data = data.iloc[:-1]

    scaler = MinMaxScaler(feature_range=(-1,1))
    data = scaler.fit_transform(data.to_numpy())

    return data, scaler


def split_data(data):
    X = data[:, 1:]
    y = data[:, 0]
    return (X,y)

def create_training_data(train_data, test_data):
    X = train_data
    y = test_data
    split_index = int(len(X) * 0.8)

    # splitting the data
    X_train, X_test = np.array(X[:split_index]), np.array(X[split_index:])
    y_train, y_test = np.array(y[:split_index]), np.array(y[split_index:])
    num_of_features = X.shape[1]
    X_train = X_train.reshape((-1, num_of_features, 1))
    X_test = X_test.reshape((-1, num_of_features, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # creating tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return (X_train, y_train, X_test, y_test)

def train(X_train, y_train, X_test, y_test):
    # creating dataset loader
    batch_size = 16
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test),
                             batch_size=batch_size,
                             shuffle=False)
    
    # defining model
    model = RNN(1, 4, 1)
    model.to(device)

    # defining hyperparameters
    learning_rate = 0.001
    num_epochs = 3
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # training step
        model.train(True)

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation step
        model.train(False)

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)

    return model

def evaluate_model(ticker:str, lookback:int, scaler, model, X_test, y_test):
    # calculating predicted price
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], X_test.shape[1]+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])
    # calculating mse
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    predictions_np = predictions.detach().numpy()

    # predicted value and mse
    predicted_price = test_predictions[-1]
    mse = float(mean_squared_error(y_test, predictions_np))

    update_result_stock(ticker,"rnn",predicted_price,mse)

def main(ticker):
    lookback = 20
    data = download_stock_data(ticker,"1h")
    data, scaler = preprocessing(data,lookback)
    train_data, test_data = split_data(data)

    X_train, y_train, X_test, y_test = create_training_data(train_data, test_data)
    model = train(X_train, y_train, X_test, y_test)
    evaluate_model(ticker,lookback,scaler, model,X_test,y_test)

    # saving model
    torch.save(model.state_dict(),f"output/rnn-{ticker}.pth")