from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from data import download_stock_data, update_db_stock, pd, np
from torch import nn
import torch


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def preprocessing(data, sequence_length:int):
    data = data.reset_index()
    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])

    data = dc(data)
    data.set_index('Date', inplace=True)
    for i in range(1, sequence_length+1):
        data[f'Close(t-{i})'] = data['Close'].shift(i)
    data.dropna(inplace=True)
    data = data.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    return data, scaler

def split_data(sequences):
    X = sequences[:, 1:]
    y = sequences[:, 0]
    return (X,y)

def create_training_data(train_data, test_data,lookback):
    X = dc(np.flip(train_data, axis=1))
    y = test_data
    split_index = int(len(X) * 0.8)

    # Prepare training data
    X_train = X[:split_index].reshape((-1, lookback, 1))
    y_train = y[:split_index].reshape((-1, 1))

    # Prepare testing data
    X_test = X[split_index:].reshape((-1, lookback, 1))
    y_test = y[split_index:].reshape((-1, 1))

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
    model = LSTM(1, 4, 1)
    model.to(device)

    # defining hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # training step
        model.train(True)
        #print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                '''
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                '''
                running_loss = 0.0

        # validation step
        model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

    return model

def evaluate_model(ticker:str, lookback:int, scaler, model, X_test, y_test):
    # calculating predicted price
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    # calculating mse
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    predictions_np = predictions.detach().numpy()

    # predicted value and mse
    predicted_price = dc(dummies[:, 0])[-1]
    mse = float(mean_squared_error(y_test, predictions_np))

    update_db_stock(ticker,"lstm",predicted_price,mse)

def main(ticker):
    lookback = 10
    data = download_stock_data(ticker)
    sequences,scaler = preprocessing(data,lookback)
    train_data, test_data = split_data(sequences)

    X_train, y_train, X_test, y_test = create_training_data(train_data, test_data,lookback)
    model = train(X_train, y_train, X_test, y_test)
    evaluate_model(ticker,lookback,scaler, model,X_test,y_test)

    # saving model
    torch.save(model.state_dict(),f"../models/{ticker}.pth")