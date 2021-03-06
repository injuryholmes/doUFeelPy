from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
# 'Agg' backend is for writing to file, not for rendering in a window.
matplotlib.use('Agg')
from IPython import embed

import matplotlib.pyplot as plt

class Sequence(nn.Module):
    # super() in python 2 version: super(Sequence, self).__init__() 
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    # super() in python3 version
    # def __init__(self):
    #     super().__init__()
    #     self.lstm1 = nn.LSTMCell(1, 51)
    #     self.lstm2 = nn.LSTMCell(51, 51)
    #     self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        # chunk完了之后就是一个tuple,len=999，每个element是一个[97,1]的tensor
        # batch size = 97
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # 调用LSTMCell()的时候，实际上最后调用的LSTMCell.forward()函数，并在内部解决了tracking和hooks
            # input: tensor containing input features
            # h_t: tensor containing the hidden state at time t for each element in the batch.
            # c_t: tensor containing the cell state at time t for each element in the batch.
            # embed()
            # h_t: [97,51]
            # c_t: [97, 51]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            # 上一个lstmCell的hidden state h_t作为下一个lstmCell的input 
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) 
            # 第二个lstm2的hidden state 作为下一个linear layer的input
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    # 每一秒都产生了100个数据，产生了999秒组数据
    input = torch.from_numpy(data[3:, :-1]) # train data input; [97,999]
    target = torch.from_numpy(data[3:, 1:]) # train data labels; [97,999]
    test_input = torch.from_numpy(data[:3, :-1]) # test data input; [3,999]
    test_target = torch.from_numpy(data[:3, 1:]) # test data labels; [3,999]
    # build the model
    seq = Sequence()
    seq.double()
    # Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input xx and target yy .
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad() # zero_grad(): Sets gradients of all model parameters to zero.
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad(): # no_grad() set the requires_grad=False
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
