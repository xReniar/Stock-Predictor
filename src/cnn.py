from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data import download_stock_data, update_result_stock, pd, np
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class StockPredictorCNN(nn.Module):
    def __init__(self):
        super(StockPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(64 * 9, 50)  # 64 filters and 9 (sequence_length - kernel_size + 1)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocessing(data: pd.Series, sequence_length: int) -> list:
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length].values
        target = data.iloc[i + sequence_length]
        sequences.append((sequence, target))
    return sequences

def split_data(sequences):
    return train_test_split(sequences, test_size=0.2, shuffle=False)

def create_training_data(train_data, test_data):
    # Prepare training data
    X_train = np.array([item[0] for item in train_data])
    y_train = np.array([item[1] for item in train_data])

    # Prepare testing data
    X_test = np.array([item[0] for item in test_data])
    y_test = np.array([item[1] for item in test_data])

    # Reshape for CNN input
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return (X_train, y_train, X_test, y_test)

def train(X_train, y_train, epochs=50, batch_size=32):
    model = StockPredictorCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_model(ticker: str, model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        last_sequence = torch.tensor(X_test[-1].reshape(1, 1, -1), dtype=torch.float32)
        predicted_price = model(last_sequence).item()
        predictions = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
        mse = mean_squared_error(y_test, predictions)
    
        update_result_stock(ticker, "cnn", predicted_price, mse)

def main(ticker):
    lookback = 10
    data = download_stock_data(ticker, "1d")['Close']
    sequences = preprocessing(data, lookback)
    train_data, test_data = split_data(sequences)

    X_train, y_train, X_test, y_test = create_training_data(train_data, test_data)
    model = train(X_train, y_train)
    evaluate_model(ticker, model, X_test, y_test)

    # saving model
    joblib.dump(model,f"output/{ticker}.pkl")

main("AAPL")