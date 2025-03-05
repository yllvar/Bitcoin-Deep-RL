from models import LinearQNet
from agent import Agent
import pandas as pd
import torch


if __name__ == '__main__':

    data = pd.read_csv('data/train_data.csv')
    data.set_index(data.columns[0], inplace=True)
    market_price = data[data.columns[0]].values
    data_test = pd.read_csv('data/test_data.csv')
    data_test.set_index(data_test.columns[0], inplace=True)
    market_price_test = data_test[data_test.columns[0]].values
    initial_money = 10000
    short_window_size = 8
    long_window_size = 22
    lag = 10
    skip = 1
    batch_size = 128
    lr = 0.01

    agent = Agent(short_window_size=short_window_size,
                  long_window_size=long_window_size,
                  lag=lag,
                  market_price=market_price,
                  skip=skip,
                  batch_size=batch_size,
                  lr=lr)
    input_size, output_size = agent.train(iterations=1000, initial_money=initial_money)

    PATH = 'models/model.pth'
    model = LinearQNet(input_size, 256, output_size)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    agent_test = Agent(
        short_window_size=short_window_size,
        long_window_size=long_window_size,
        lag=lag,
        market_price=market_price_test,
        skip=skip,
        batch_size=batch_size,
        lr=lr)

    agent_test.test(initial_money=initial_money, model=model, epsilon=-1)
