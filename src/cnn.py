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
    
class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * (input_channels // 2), 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_features, 1) -> (batch_size, 1, num_features)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocessing(data: pd.DataFrame, lookback: int):
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop(columns=["Adj Close"])

    # adding indicator values
    indicators.sma(data, lookback)
    indicators.std(data, lookback)
    indicators.bb(data, lookback)
    indicators.ema(data, lookback)
    indicators.rsi(data, lookback)

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

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.to_numpy())

    return data, scaler

def split_data(data):
    X = data[:, 1:]
    y = data[:, 0]
    return (X, y)

def create_training_data(train_data, test_data):
    X = train_data
    y = test_data
    split_index = int(len(X) * 0.8)

    # splitting the data
    X_train, X_test = np.array(X[:split_index]), np.array(X[split_index:])
    y_train, y_test = np.array(y[:split_index]), np.array(y[split_index:])
    num_of_features = X.shape[1]
    X_train = X_train.reshape((-1, 1, num_of_features))
    X_test = X_test.reshape((-1, 1, num_of_features))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # creating tensors
    X_train = torch.tensor(X_train, device=device).float()
    y_train = torch.tensor(y_train, device=device).float()
    X_test = torch.tensor(X_test, device=device).float()
    y_test = torch.tensor(y_test, device=device).float()

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
    num_features = X_train.shape[2]
    model = CNN(1, 1)
    model.to(device)

    # defining hyperparameters
    learning_rate = 0.001
    num_epochs = 10
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

def evaluate_model(ticker: str, lookback: int, scaler, model, X_test: torch.Tensor, y_test: torch.Tensor):
    # calculating predicted price
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], X_test.shape[2] + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])
    # calculating mse
    model.eval()
    with torch.no_grad():
        predictions: torch.Tensor = model(X_test)
    predictions_np = predictions.detach().cpu().numpy()

    # predicted value and mse
    predicted_price = test_predictions[-1]
    mse = float(mean_squared_error(y_test.cpu(), predictions_np))

    update_result_stock(ticker, "cnn", predicted_price, mse)

def main(ticker):
    lookback = 20
    data = download_stock_data(ticker, "1d")
    data, scaler = preprocessing(data, lookback)
    train_data, test_data = split_data(data)

    X_train, y_train, X_test, y_test = create_training_data(train_data, test_data)
    model = train(X_train, y_train, X_test, y_test)
    evaluate_model(ticker, lookback, scaler, model, X_test, y_test)

    # saving model
    torch.save(model.state_dict(), f"output/cnn-{ticker}.pth")
