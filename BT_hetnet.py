"""
This is the simulation code for the DL models in the HetNet scenario in the paper "LSTM-Based Predictive mmWave Beam Tracking via Sub-6 GHz Channels for V2I Communications"
"""
import torch, gc
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from model_fusion import MLP_fusion, LSTM_fusion, CNN_fusion, MLP, LSTM
import scipy.io as sio
from math_models import trans_rate
import random
# Load data
Sub6G_channel = sio.loadmat('data/SUB_16MM/SUB6_16')['nn_input']  # Sequential Sub-6 GHz CSI at previous 40 time slots and present
mm_beam = sio.loadmat('data/SUB_16MM/SUB6_16')['nn_label'] # The mmWave optimal beam at present
mm_channel = sio.loadmat('data/SUB_16MM/SUB6_16')['mm_channel']    # The mmWave channels at present
mm_input = sio.loadmat('data/SUB_16MM/SUB6_16')['mm_input']    # Sequential mmWave wide beam measurement at previous 40 time slots and present
bf_matrix = sio.loadmat('data/co_located/data_v20')['W_tx'] # The beamforming codebook of 8*8 antenna array

# Hyper Parameters
EPOCH = 10               # train the training data n times
BATCH_SIZE = 64
TIME_STEP = 32          # rnn time step 
BS_N = 16
FEATURE_SIZE = 8         # feature size 
FEATURE_SIZE_MM = 4 * 2 * BS_N         # feature size mmWave
WIDE_BEAM = 16         # selected wide beams 
HIDDEN_SIZE = [64, 64, 64]        # rnn hidden size 
ANN_NUM = 64
CLASS_NUM = ANN_NUM * BS_N         # number of classes
HIDDEN_LAYERS = 2       # rnn hidden layer numbers 
DROPOUT = 0
BN = True
LR = 0.01               # learning rate

# Communication parameter
B = 500 # bandwidth = 500 MHz
TX_POWER = 23   # 23 dBm
NOISE_POWER = -174  # -174 dBm/Hz

# Data preprocessing
X = Sub6G_channel.transpose(2, 1, 0)[:, -TIME_STEP-2:-1, :]
X = X.reshape(X.shape[0], -1)
scaler = StandardScaler()

x_std = X #scaler.fit_transform(X)

r_std = mm_input.transpose(2, 1, 0)[:, -TIME_STEP-1:-1, :] #scaler.fit_transform(mm_input.transpose())
r_std = r_std.reshape(r_std.shape[0], -1)

x_input = np.hstack((x_std, r_std))
Y = np.vstack((mm_beam-1, mm_channel)).transpose(1, 0)

def train_net(net, optimizer, loss_func, train_data_sub, train_data_mm, train_label, fusion=True):
    net.train()
    if fusion:
        out_sub, out_mm, output  = net(train_data_sub, train_data_mm)
        loss = loss_func(out_sub, train_label) + loss_func(out_mm, train_label) + loss_func(output, train_label)
    else:
        out_sub  = net(train_data_sub)
        loss = loss_func(out_sub, train_label)
    optimizer.zero_grad()                         # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()

    return loss

def test_net(net, test_data_sub, test_data_mm, test_label, W_tx, fusion=True):
    net.eval()
    gc.collect()
    torch.cuda.empty_cache()
    if fusion:
        out_sub, out_mm, test_output = net(test_data_sub, test_data_mm)                   # (samples, time_step, input_size)
    else:
        test_output = net(test_data_sub)
    test_num = test_label.shape[0]
    beam_label = np.int64(test_label[:,0].real)
    channel =  test_label[:, 1:].reshape(test_num, BS_N, ANN_NUM)
    best_bs = beam_label // ANN_NUM
    best_beam = beam_label % ANN_NUM
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    pred_bs = pred_y // ANN_NUM
    pred_beam = pred_y % ANN_NUM
    pred_y_3 = torch.topk(test_output, 3, 1)[1].data.cpu().numpy()
    pred3_bs = pred_y_3 // ANN_NUM
    pred3_beam = pred_y_3 % ANN_NUM
    accuracy = float((pred_y == beam_label).astype(int).sum())# / float(test_num)
    accuracy_3 = 0.0
    best_rate = np.zeros(test_num)
    test_rate = np.zeros(test_num)
    test_rate_3 = np.zeros((test_num, 3))
    rand_rate = np.zeros(test_num)
    for i in range(test_num):
        accuracy_3 += (pred_y_3[i] == beam_label[i]).astype(int).sum()
        best_rate[i] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i,best_bs[i],:], W_tx[:,best_beam[i]])
        test_rate[i] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i,pred_bs[i],:], W_tx[:,pred_beam[i]])
        for j in range(3):
            test_rate_3[i][j] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i,pred3_bs[i][j],:], W_tx[:,pred3_beam[i][j]])
        rand_bs = random.randrange(1, BS_N, 1)
        rand_beam = random.randrange(1, ANN_NUM, 1)
        rand_rate[i] = trans_rate(B, TX_POWER, NOISE_POWER, channel[i,rand_bs,:], W_tx[:,rand_beam])
    
    accuracy_3 = float(accuracy_3)# / float(test_num)
    rate_3 = np.max(test_rate_3, axis=1)
    
    return accuracy, accuracy_3, best_rate, test_rate, rate_3, rand_rate


X_train, X_test, y_train, y_test = train_test_split(x_input, Y, random_state=4, shuffle=True)

test_x_sub = torch.from_numpy(X_test[:, :(TIME_STEP+1)*FEATURE_SIZE]).view(-1, TIME_STEP+1, FEATURE_SIZE)   
test_x_mm = torch.from_numpy(X_test[:, (TIME_STEP+1)*FEATURE_SIZE:]).view(-1, TIME_STEP, FEATURE_SIZE_MM)  


mlp_mm = MLP(FEATURE_SIZE_MM, HIDDEN_SIZE[1], CLASS_NUM, DROPOUT, BN)
mlp = MLP_fusion(FEATURE_SIZE, FEATURE_SIZE_MM, WIDE_BEAM, [128, 128, 128], CLASS_NUM, DROPOUT, BN)
lstm = LSTM_fusion(FEATURE_SIZE, FEATURE_SIZE_MM, WIDE_BEAM, HIDDEN_SIZE, CLASS_NUM, HIDDEN_LAYERS, DROPOUT, BN)
lstm_only = LSTM(FEATURE_SIZE, HIDDEN_SIZE[0], CLASS_NUM, 2, DROPOUT, BN)
cnn = CNN_fusion(FEATURE_SIZE, FEATURE_SIZE_MM, WIDE_BEAM, [64, 128], CLASS_NUM, HIDDEN_LAYERS, DROPOUT, BN)
if torch.cuda.is_available():
    mlp_mm = mlp_mm.cuda()
    mlp = mlp.cuda()
    lstm = lstm.cuda()
    cnn = cnn.cuda()
    lstm_only = lstm_only.cuda()
    test_x_sub = test_x_sub.cuda()
    test_x_mm = test_x_mm.cuda()

optimizer_mlp_m = torch.optim.Adam(mlp_mm.parameters(), lr=LR)   
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=LR)  
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=LR)
optimizer_lstm_only = torch.optim.Adam(lstm_only.parameters(), lr=LR)
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()                      
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

# training and testing 
for epoch in range(EPOCH):
    for step in range(X_train.shape[0] // BATCH_SIZE):   # gives batch data       
        b_x = torch.from_numpy(X_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE])              
        b_x_sub = b_x[:, :(TIME_STEP+1)*FEATURE_SIZE].view(-1, TIME_STEP+1, FEATURE_SIZE)   # reshape x to (batch, time_step, input_size)
        b_x_mm = b_x[:, (TIME_STEP+1)*FEATURE_SIZE:].view(-1, TIME_STEP, FEATURE_SIZE_MM)
        b_y = torch.from_numpy(np.int64(y_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE, 0].real))   
        if torch.cuda.is_available():
            b_x_sub = b_x_sub.cuda()
            b_x_mm = b_x_mm.cuda()
            b_y = b_y.cuda()
                                                
        
        loss_mlp_mm = train_net(mlp_mm, optimizer_mlp_m, loss_func, b_x_mm[:, -1, :], b_x_mm[:, -1, :], b_y, False)    

        loss_mlp = train_net(mlp, optimizer_mlp, loss_func, b_x_sub[:, -2, :], b_x_mm[:, -1, :], b_y)

        loss_lstm = train_net(lstm, optimizer_lstm, loss_func, b_x_sub[:, :-1, :], b_x_mm[:, -1, :], b_y)

        loss_cnn = train_net(cnn, optimizer_cnn, loss_func, b_x_sub[:, -2, :], b_x_mm[:, -1, :], b_y)

        loss_lstm_only = train_net(lstm_only, optimizer_lstm_only, loss_func, b_x_sub[:, 1:, :], b_x_mm[:, -1, :], b_y, False)

        gc.collect()
        torch.cuda.empty_cache()
        
            
    if epoch % 10 == 9:
        accuracy, accuracy_3, best_rate, test_rate_mlpmm, test_rate_3_mlpmm, rand_rate_mlpmm = test_net(mlp_mm, test_x_mm[:, -1, :], test_x_mm[:, -1, :], y_test, bf_matrix, False)
        print('Epoch: ', epoch, '| mlp_m train loss: %.4f' % loss_mlp_mm.data.cpu().numpy(), '| mlp_m test accuracy: %.2f' % accuracy, '| mlp_m test accuracy_3: %.2f' % accuracy_3, 
                '| mlp_m test rate: %.2f' % test_rate_mlpmm.mean(), '| mlp_m test rate_3: %.2f' % test_rate_3_mlpmm.mean())

        accuracy, accuracy_3, best_rate, test_rate_mlp, test_rate_3_mlp, rand_rate_mlp = test_net(mlp, test_x_sub[:, -2, :], test_x_mm[:, -1, :], y_test, bf_matrix)
        print('Epoch: ', epoch, '| mlp train loss: %.4f' % loss_mlp.data.cpu().numpy(), '| mlp test accuracy: %.2f' % accuracy, '| mlp test accuracy_3: %.2f' % accuracy_3, 
                '| mlp test rate: %.2f' % test_rate_mlp.mean(), '| mlp test rate_3: %.2f' % test_rate_3_mlp.mean())

        accuracy, accuracy_3, best_rate, test_rate_lstm, test_rate_3_lstm, rand_rate_lstm = test_net(lstm, test_x_sub[:, :-1, :], test_x_mm[:, -1, :], y_test, bf_matrix)
        print('Epoch: ', epoch, '| lstm train loss: %.4f' % loss_lstm.data.cpu().numpy(), '| lstm test accuracy: %.2f' % accuracy, '| lstm test accuracy_3: %.2f' % accuracy_3, 
                '| lstm test rate: %.2f' % test_rate_lstm.mean(), '| lstm test rate_3: %.2f' % test_rate_3_lstm.mean())

        accuracy, accuracy_3, best_rate, test_rate_cnn, test_rate_3_cnn, rand_rate_cnn = test_net(cnn, test_x_sub[:, -2, :], test_x_mm[:, -1, :], y_test, bf_matrix)
        print('Epoch: ', epoch, '| cnn train loss: %.4f' % loss_cnn.data.cpu().numpy(), '| cnn test accuracy: %.2f' % accuracy, '| cnn test accuracy_3: %.2f' % accuracy_3, 
                '| cnn test rate: %.2f' % test_rate_cnn.mean(), '| cnn test rate_3: %.2f' % test_rate_3_cnn.mean())
        
        accuracy, accuracy_3, best_rate, test_rate_lstm_o, test_rate_3_lstm_o, rand_rate_lstm_o = test_net(lstm_only, test_x_sub[:, 1:, :], test_x_mm[:, -1, :], y_test, bf_matrix, False)
        print('Epoch: ', epoch, '| lstm_only train loss: %.4f' % loss_lstm_only.data.cpu().numpy(), '| lstm_only test accuracy: %.2f' % accuracy, '| lstm_only test accuracy_3: %.2f' % accuracy_3, 
                '| best rate: %.2f' % best_rate.mean(), '| lstm_only test rate: %.2f' % test_rate_lstm_o.mean(), '| lstm_only test rate_3: %.2f' % test_rate_3_lstm_o.mean(), '| rand rate: %.2f' % rand_rate_lstm_o.mean())
        