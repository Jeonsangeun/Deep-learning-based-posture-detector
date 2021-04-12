import collections
import random as rd
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

start_time = time.time()

learning_rate = 0.001
mini_batch = 200

sample_size = 200

data_train = np.load('train_data2.npy')
data_test = np.load('posture1.npy')

test_data = np.load('test_data.npy')

input_size = data_train.shape[2]
output_size = data_test.size

print(data_train.shape)

loss_label = []
train_corr_label = []
test_corr_label = []

class Buffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=5000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        batch = rd.sample(self.buffer, n)
        data_lst, posture_lst = [], []

        for transition in batch:
            d, p = transition
            data_lst.append(d)
            posture_lst.append(p)

        return torch.tensor(data_lst, dtype=torch.float), torch.tensor(posture_lst, dtype=torch.long)

    def size(self):
        return len(self.buffer)

class DNN_net(nn.Module):
    def __init__(self):
        super(DNN_net, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, 240)
        self.fc3 = nn.Linear(240, 120)
        self.fc4 = nn.Linear(120, 60)
        self.fc5 = nn.Linear(60, output_size)

        # self.activate_fun = nn.ReLU(inplace=True)
        # self.activate_fun = nn.LeakyReLU(inplace=True)
        # self.activate_fun = nn.Tanh()
        # self.activate_fun = nn.Softmax()
        self.activate_fun = nn.ELU(inplace=True)
        self.fc_module = nn.Sequential(self.fc1, self.activate_fun, self.fc2, self.activate_fun,
                                        self.fc3, self.activate_fun, self.fc4, self.activate_fun, self.fc5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc_module(x)
        return out

model = DNN_net()

criterion = nn.CrossEntropyLoss()
# criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def Train(memory, model):

    data, posture = memory.sample(mini_batch)
    output = model(data)
    # print(output)
    # print(posture.shape)

    loss = criterion(output, posture)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def Test(model, memory, n):
    correct = 0
    data, posture = memory.sample(n)
    for i in range(n):
        with torch.no_grad():
            output = model(data[i].unsqueeze(0))
        target = posture[i].unsqueeze(0)

        # loss += criterion(output, target)

        pred = output.data.max(1, keepdim=True)[1][0]

        if pred == target:
            correct += 1
    correct = correct / 100
    return correct

def Test_simulation(model):
    correct = 0
    data_N = test_data.shape[1]
    posture = torch.from_numpy(data_test).int()
    for i in range(output_size):
        for k in range(data_N):
            state = torch.from_numpy(test_data[i, k]).float().unsqueeze(0)
            with torch.no_grad():
                output = model(state)

            target = posture[i].unsqueeze(0)
            pred = output.data.max(1, keepdim=True)[1][0]
            if pred == target:
                correct += 1

    correct = correct / (output_size * data_N)
    return 100 * correct

def main():
    max_episode = 2001
    loss_avg = 0
    memory = Buffer()
    start_time = time.time()
    for episode in range(max_episode):
        for i in range(mini_batch):
            k = np.random.randint(output_size)
            m = np.random.randint(sample_size)
            data = data_train[k, m]
            posture = data_test[k]
            memory.put((data, posture))
        loss = Train(memory, model.eval())
        loss_avg += loss

        if episode % 10 == 0 and not episode == 0:
            train_corr = Test(model.eval(), memory, 100)
            test_corr = Test_simulation(model.eval())
            print('===========================\n_Train set: Accuracy: {}, loss: {:5f}'.
                  format(100.0*train_corr, loss_avg / 10.0))
            print('Test set: Accuracy: {}'.format(test_corr))

            train_corr_label.append(100.0*train_corr)
            test_corr_label.append(test_corr)
            loss_label.append(loss_avg / 10.0)

            loss_avg = 0

            if test_corr == 100.0 and 100.0*train_corr == 100.0 and loss <= 0.01:
                torch.save(model.state_dict(), "save_ELU2.pth")

    print("start_time", start_time)
    print("--- %s seconds ---" % (time.time() - start_time))

    x_label = np.array(range(200))
    plt.subplot(1, 2, 1)
    plt.plot(x_label, train_corr_label, ':', lw=2, label='train_accuracy')
    plt.plot(x_label, test_corr_label, '--', lw=2, label='test_accuracy')
    plt.ylim(0, 100)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy [%]", fontsize=13)
    plt.legend(loc="lower right")
    plt.subplot(1, 2, 2)
    plt.plot(x_label, loss_label, '-.', lw=2, label='loss_accuracy')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=13)
    plt.ylim(0, 10)
    # plt.xlim(0, 130000)

    plt.legend(loc="upper right")
    plt.show()

    # np.save("Leaky", test_corr_label)

if __name__ == '__main__':
    main()


