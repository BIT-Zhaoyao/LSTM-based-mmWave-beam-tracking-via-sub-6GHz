"""
This is the simulation code for the DL models in the co-located scenario in the paper "LSTM-Based Predictive mmWave Beam Tracking via Sub-6 GHz Channels 
for V2I Communications"
"""
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from model import MLP, LSTM, CNN, CNN_LSTM
import scipy.io as sio
from math_models import trans_rate
import random
Sub6G_channel = sio.loadmat('data/co_located/data_v20')['nn_input']  # Sequential Sub-6 GHz CSI at previous 100 time slots and present
mm_beam = sio.loadmat('data/co_located/data_v20')['nn_label']   # The mmWave optimal beam at present
mm_channel = sio.loadmat('data/co_located/data_v20')['mm_channel']  # The mmWave channels at present
bf_matrix = sio.loadmat('data/co_located/data_v20')['W_tx'] # The beamforming codebook of 8*8 antenna array

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20               # train the training data n times
BATCH_SIZE = 64
TIME_STEP = 16          # time step 
FEATURE_SIZE = 8         # feature size 
HIDDEN_SIZE = 64        # hidden size 
CLASS_NUM = 64          # number of classes
HIDDEN_LAYERS = 2       # rnn hidden layer numbers 
DROPOUT = 0
BN = False
LR = 0.01               # learning rate

# Communication parameter
B = 500 # bandwidth = 500 MHz
TX_POWER = 23   # 23 dBm
NOISE_POWER = -174  # -174 dBm/Hz

# Data preprocessing
X = Sub6G_channel.transpose(2, 1, 0)[:, -TIME_STEP-1:-1, :]
X = X.reshape(X.shape[0], -1)
scaler = StandardScaler()
scaler.fit(X)

x_std = scaler.transform(X)
Y = np.vstack((mm_beam-1, mm_channel)).transpose(1, 0)
X_train, X_test, y_train, y_test = train_test_split(x_std, Y, train_size=0.75, test_size=0.25, random_state=4, shuffle=True)

test_x = torch.from_numpy(X_test).view(-1, TIME_STEP, FEATURE_SIZE)   

mlp = MLP(FEATURE_SIZE, [128, 256], CLASS_NUM, HIDDEN_LAYERS, DROPOUT, BN)
lstm = LSTM(FEATURE_SIZE, HIDDEN_SIZE, CLASS_NUM, HIDDEN_LAYERS, DROPOUT, BN)
cnn = CNN(FEATURE_SIZE, [128, 128], CLASS_NUM, HIDDEN_LAYERS, DROPOUT, BN)
cnn_lstm = CNN_LSTM(FEATURE_SIZE, HIDDEN_SIZE, CLASS_NUM, HIDDEN_LAYERS, DROPOUT, BN)
if torch.cuda.is_available():
    mlp = mlp.cuda()
    lstm = lstm.cuda()
    cnn = cnn.cuda()
    cnn_lstm = cnn_lstm.cuda()
    test_x = test_x.cuda()

optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=LR)#, weight_decay=1e-4)   # optimize all cnn parameters
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=LR)#, weight_decay=1e-4)   # optimize all cnn parameters
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=LR)#, weight_decay=1e-4)   # optimize all cnn parameters
optimizer_cnn_lstm = torch.optim.Adam(cnn_lstm.parameters(), lr=LR)#, weight_decay=1e-4)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
if torch.cuda.is_available():
    loss_func = loss_func.cuda()


def train_net(net, optimizer, loss_func, train_data, train_label):
    net.train()
    output = net(train_data)
    loss = loss_func(output, train_label)
    optimizer.zero_grad()                         # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()

    return loss

def test_net(net, test_data, test_label, W_tx):
    net.eval()
    test_output = net(test_data)                   # (samples, time_step, input_size)
    test_num = test_label.shape[0]
    beam_label = np.int64(test_label[:,0].real)
    channel =  test_label[:, 1:]
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    pred_y_3 = torch.topk(test_output, 3, 1)[1].data.cpu().numpy()
    accuracy = float((pred_y == beam_label).astype(int).sum()) / float(test_num)
    accuracy_3 = 0.0
    best_rate = np.zeros(test_num)
    test_rate = np.zeros(test_num)
    test_rate_3 = np.zeros((test_num, 3))
    rand_rate = np.zeros(test_num)
    for i in range(test_num):
        accuracy_3 += (pred_y_3[i] == beam_label[i]).astype(int).sum()
        best_rate[i] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i], W_tx[:,beam_label[i]])
        test_rate[i] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i], W_tx[:,pred_y[i]])
        for j in range(3):
            test_rate_3[i][j] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i], W_tx[:,pred_y_3[i][j]])

        rand_beam = random.randrange(1, CLASS_NUM, 1)
        rand_rate[i] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i], W_tx[:,0])

    
    accuracy_3 = float(accuracy_3) / float(test_num)
    rate_3 = np.max(test_rate_3, axis=1)
    
    return accuracy, accuracy_3, best_rate, test_rate, rate_3, rand_rate

# training and testing
for epoch in range(EPOCH):
    for step in range(X_train.shape[0] // BATCH_SIZE):   # gives batch data
        b_x = torch.from_numpy(X_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE])              
        b_x = b_x.view(-1, TIME_STEP, FEATURE_SIZE)   # reshape x to (batch, time_step, input_size)
        b_y = torch.from_numpy(np.int64(y_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE, 0].real))    
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        # Calculate the computation complex
        # from thop import profile
        # flops, params = profile(mlp, inputs=(b_x[1, -1, :].unsqueeze(0), ))

        loss_mlp = train_net(mlp, optimizer_mlp, loss_func, b_x[:, -1, :], b_y)

        loss_lstm = train_net(lstm, optimizer_lstm, loss_func, b_x, b_y)

        loss_cnn = train_net(cnn, optimizer_cnn, loss_func, b_x[:, -1, :], b_y)

        loss_cnn_lstm = train_net(cnn_lstm, optimizer_cnn_lstm, loss_func, b_x, b_y)


    accuracy, accuracy_3, best_rate, test_rate_mlp, test_rate_3_mlp, rand_rate_mlp = test_net(mlp, test_x[:, -1, :], y_test, bf_matrix)
    print('Epoch: ', epoch, '| mlp train loss: %.4f' % loss_mlp.data.cpu().numpy(), '| mlp test accuracy: %.2f' % accuracy, '| mlp test accuracy_3: %.2f' % accuracy_3, 
        '| mlp test rate: %.2f' % test_rate_mlp.mean(), '| mlp test rate_3: %.2f' % test_rate_3_mlp.mean())

    accuracy, accuracy_3, best_rate, test_rate_lstm, test_rate_3_lstm, rand_rate_lstm = test_net(lstm, test_x, y_test, bf_matrix)
    print('Epoch: ', epoch, '| lstm train loss: %.4f' % loss_lstm.data.cpu().numpy(), '| lstm test accuracy: %.2f' % accuracy, '| lstm test accuracy_3: %.2f' % accuracy_3, 
         '| lstm test rate: %.2f' % test_rate_lstm.mean(), '| lstm test rate_3: %.2f' % test_rate_3_lstm.mean())
    
    accuracy, accuracy_3, best_rate, test_rate_cnn, test_rate_3_cnn, rand_rate_cnn = test_net(cnn, test_x[:, -1, :], y_test, bf_matrix)
    print('Epoch: ', epoch, '| cnn train loss: %.4f' % loss_cnn.data.cpu().numpy(), '| cnn test accuracy: %.2f' % accuracy, '| cnn test accuracy_3: %.2f' % accuracy_3, 
        '| cnn test rate: %.2f' % test_rate_cnn.mean(), '| cnn test rate_3: %.2f' % test_rate_3_cnn.mean())
    
    accuracy, accuracy_3, best_rate, test_rate_cnnl, test_rate_3_cnnl, rand_rate_cnnl = test_net(cnn_lstm, test_x, y_test, bf_matrix)
    print('Epoch: ', epoch, '| cnn_l train loss: %.4f' % loss_cnn_lstm.data.cpu().numpy(), '| cnn_l test accuracy: %.2f' % accuracy, '| cnn_l test accuracy_3: %.2f' % accuracy_3, 
        '| cnn_l best rate: %.2f' % best_rate.mean(), '| cnn_l test rate: %.2f' % test_rate_cnnl.mean(), '| cnn_l test rate_3: %.2f' % test_rate_3_cnnl.mean(), '| cnn_l rand rate_3: %.2f' % rand_rate_cnnl.mean())

torch.save({
        'mlp_para': mlp.state_dict(),
        'cnn_para': cnn.state_dict(),
        'lstm_para': lstm.state_dict(),
        'cnn_lstm_para': cnn_lstm.state_dict(),
    }, 'co_located_v20.pth')

