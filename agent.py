from sklearn.preprocessing import MinMaxScaler
from models import LinearQNet, QTrainer
from collections import deque
import numpy as np
import random
import torch
import tqdm
import copy
import logging

logging.basicConfig(level=logging.INFO)


class Agent:
    def __init__(self,
                 short_window_size,
                 long_window_size,
                 lag,
                 market_price,
                 skip,
                 batch_size,
                 lr):
        self.short_window_size = short_window_size
        self.long_window_size = long_window_size
        self.window_size = long_window_size + lag
        self.lag = lag
        self.market_price = market_price
        self.open_fee = 0.0004
        self.close_fee = 0.0004
        self.skip = skip
        self.action_size = 3
        self.future_day = 1
        self.batch_size = batch_size
        self.train_size = 2000
        self.input_size = lag * 3
        self.memory = deque(maxlen=self.train_size)
        self.short_memory = deque(maxlen=self.batch_size)
        self.gamma = 0.95
        self.epsilon_greedy = True
        self.epsilon = 0.5
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.lr = lr

        self.model = LinearQNet(self.input_size, 256, self.action_size)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

    @staticmethod
    def __scaler(data):
        scaler = MinMaxScaler(feature_range=(1, 2))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        return scaled_data.reshape(-1)

    def act(self, state, epsilon):
        if self.epsilon_greedy == 1:
            if random.random() <= epsilon:
                return random.randrange(self.action_size)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()
        else:
            if random.random() <= self.epsilon:
                return random.randrange(self.action_size)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()

    def get_state(self, t):
        window_size = self.window_size
        d = t - window_size
        state = np.array([])
        if d >= 0:
            window_of_data = self.market_price[d: t]
        else:
            window_of_data = np.concatenate([-d * [self.market_price[0]], self.market_price[1: t + 1]])
        start_point_of_short_ma = len(window_of_data) - self.short_window_size - self.lag
        for counter in np.arange(0, self.lag):
            long_ma = window_of_data[counter:self.long_window_size + counter]
            short_ma = window_of_data[counter + start_point_of_short_ma:
                                      start_point_of_short_ma + self.short_window_size + counter]
            state = np.append(state, np.mean(short_ma))
            state = np.append(state, np.mean(long_ma))
            state = np.append(state, np.mean(short_ma) - np.mean(long_ma))

        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self):
        states, actions, rewards, next_states, dones = zip(*self.short_memory)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train(self, iterations, initial_money):
        self.market_price = self.__scaler(self.market_price)
        previous_profit = 0
        decay_factor = (self.epsilon_min / self.epsilon) ** (1 / iterations)
        epsilon = self.epsilon_max
        for i in tqdm.tqdm(np.arange(iterations)):
            epsilon = epsilon * decay_factor
            total_profit = 0
            inventory = np.array([])
            state = self.get_state(0)
            previous_money = initial_money
            money = initial_money
            self.memory = deque(maxlen=self.train_size)
            self.short_memory = deque(maxlen=self.batch_size)
            for t in range(0, len(self.market_price) - self.future_day - 1, self.skip):
                action = self.act(state, epsilon)
                next_state = self.get_state(t + 1)
                invest = float(0)
                if action == 1 and money > 0:
                    quantity = (money / ((1 - self.open_fee) * self.market_price[t]))
                    inventory = np.append(inventory, [quantity])
                    money_after_n_days = 0
                    money_after_n_days += (self.__close_position(quantity,
                                                                np.mean(self.market_price[t:t + self.future_day])))
                    profit = money_after_n_days - money

                    previous_money = money
                    money = 0
                    invest = float(profit)
                elif action == 2 and len(inventory) > 0:
                    final_money = 0
                    quantity = np.array(inventory.tolist().pop(0))
                    inventory = np.delete(inventory, 0)
                    final_money += self.__close_position(quantity, self.market_price[t])

                    profit = final_money - previous_money
                    money = final_money
                    total_profit += profit
                    invest = self.__reward(profit)
                    previous_money = final_money

                if random.random() <= (self.train_size/len(self.market_price)):
                    self.remember(state, action, invest, next_state,
                                t >= len(self.market_price) - self.future_day - 1)
                self.short_memory.append((state, action, invest, next_state,
                                        t >= len(self.market_price) - self.future_day - 1))

                state = next_state
                if (t + 1) % self.batch_size == 0:
                    self.train_short_memory()

            if len(inventory) > 0:
                final_money = 0
                quantity = np.array(inventory.tolist().pop(0))
                inventory = np.delete(inventory, 0)
                final_money += self.__close_position(quantity, self.market_price[t])

                profit = final_money - previous_money
                money = final_money
                total_profit += profit

            if total_profit > previous_profit:
                previous_profit = total_profit
                logging.info('epoch: %d, total profit: %f.3, total money: %f, epsilon: %f',
                            i + 1, total_profit, money, epsilon)
                torch.save(self.model.state_dict(), 'models/model.pth')

            self.train_long_memory()

        return self.input_size, self.action_size


    @staticmethod
    def __reward(profit):
        if profit > 0:
            return float(profit)
        else:
            return float(profit) * 1.5  # Penalize losses more

    def test(self, initial_money, model, epsilon):
        self.market_price = self.__scaler(self.market_price)
        self.model = copy.deepcopy(model)
        total_profit = 0
        state = self.get_state(0)
        inventory = np.array([])
        previous_money = initial_money
        final_money = initial_money
        money = initial_money
        for t in range(0, len(self.market_price), self.skip):
            action = self.act(state, epsilon)
            next_state = self.get_state(t + 1)
            if action == 1 and money > 0:
                quantity = (money / ((1 - self.open_fee) * self.market_price[t]))
                inventory = np.append(inventory, [quantity])
                money = 0
            elif action == 2 and len(inventory) > 0:
                final_money = 0
                quantity = np.array(inventory.tolist().pop(0))
                inventory = np.delete(inventory, 0)
                final_money += self.__close_position(quantity, self.market_price[t])

                profit = final_money - previous_money
                money = final_money
                total_profit += profit
                previous_money = final_money

            state = next_state

        if len(inventory) > 0:
            final_money = 0
            quantity = np.array(inventory.tolist().pop(0))
            inventory = np.delete(inventory, 0)
            final_money += self.__close_position(quantity, self.market_price[t])

            profit = final_money - previous_money
            total_profit += profit
        print('total rewards: %f.3, final money: %f' % (total_profit, final_money))

    def __close_position(self, quantity, price):
        return quantity * price * (1 - self.close_fee)

    def __open_position(self, quantity, price):
        return quantity * price * (1 - self.open_fee)
