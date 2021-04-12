import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

input_size = 240
output_size = 8

# class Dnet(nn.Module):
#     def __init__(self):
#         super(Dnet, self).__init__()
#         self.fc1 = nn.Linear(input_size, 12)
#         self.fc2 = nn.Linear(12, 12)
#         self.fc3 = nn.Linear(12, 12)
#         self.fc4 = nn.Linear(12, 12)
#         self.fc5 = nn.Linear(12, output_size)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = F.leaky_relu(self.fc3(x))
#         x = F.leaky_relu(self.fc4(x))
#         x = F.leaky_relu(self.fc5(x))
#         return x

class Dnet(nn.Module):
    def __init__(self):
        super(Dnet, self).__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, 15)
        self.fc5 = nn.Linear(15, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        return x

def main():
    W = Dnet()
    state = np.load('data.npy')
    W.load_state_dict(torch.load("save1.pth"))
    data = torch.from_numpy(state[2]).float()
    print(W(data))
    print(np.argmax(W(data).detach().numpy()))


if __name__ == '__main__':
    main()
