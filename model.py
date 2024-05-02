import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Generate a MLP cell
    """
    def __init__(self, features, hidden_size, class_num, num_layers = 2, dropout = 0, do_bn = False):
        
        super(MLP, self).__init__()
        
        self.bn_en = nn.BatchNorm1d(features, momentum=0.5)
        self.mlp = nn.Sequential(
            nn.Linear(features, hidden_size[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Dropout(dropout),
            nn.ReLU()
            )
        self.bn_de = nn.BatchNorm1d(hidden_size[1], momentum=0.5)
        self.decoder = nn.Sequential(nn.Linear(hidden_size[1], class_num), nn.Dropout(dropout))
        self.do_bn = do_bn
        
    def forward(self, x):      
        x = x.to(torch.float32)
        if self.do_bn:
            x = self.bn_en(x)
        output = self.mlp(x)
       
        if self.do_bn:
            output = self.bn_de(output)
        out = self.decoder(output)
        
        return out
  
class LSTM(nn.Module):
    """
    Generate a LSTM cell
    """
    def __init__(self, features, hidden_size, class_num, num_layers = 2, dropout = 0, do_bn = False):
        
        super(LSTM, self).__init__()
        
        self.bn_en = nn.BatchNorm1d(features, momentum=0.5)
        self.lstm = nn.LSTM(features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn_de = nn.BatchNorm1d(hidden_size, momentum=0.5)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, class_num), nn.Dropout(dropout))
        self.do_bn = do_bn
        
    def forward(self, x):
        # if len(x.shape) > 3:
            # print('x shape must be 3')
            # 
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)        
        x = x.to(torch.float32)
        B, L, F = x.shape
        output = x.reshape(B * L, -1) 
        if self.do_bn:
            output = self.bn_en(output)
        output = output.reshape(B, L, -1)       
        output, (cur_hidden, cur_cell) = self.lstm(output, None)
        
        output = output.reshape(B * L, -1)
        
        if self.do_bn:
            output = self.bn_de(output)
        output = self.decoder(output)
        output = output.reshape(B, L, -1) 
        # choose r_out at the last time step
        out = output[:, -1, :]
        
        return out


class CNN(nn.Module):
    """
    Generate a CNN cell
    """
    def __init__(self, features, hidden_size, class_num, num_layers = 2, dropout = 0, do_bn = False):
        
        super(CNN, self).__init__()
        
        self.bn_en = nn.BatchNorm1d(features, momentum=0.5)
        self.conv1 = nn.Sequential(         
            nn.Conv1d(
                in_channels=2,              # input height
                out_channels=hidden_size[0],            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding='same',                 
            ),                              
            nn.Dropout(dropout),
            nn.ReLU(),                      # activation
        )
        self.conv2 = nn.Sequential(         # input shape (64, 2)
            nn.Conv1d(hidden_size[0], hidden_size[1], 3, 1, 'same'),     # output shape (64, 2)
            nn.Dropout(dropout),
            nn.ReLU(),                      # activation
            nn.MaxPool1d(int(features / 2)),                # output shape (64, 1)
        )
        self.bn_de = nn.BatchNorm1d(hidden_size[1], momentum=0.5)
        self.decoder = nn.Sequential(nn.Linear(hidden_size[1], class_num), nn.Dropout(dropout))
        self.do_bn = do_bn
        
    def forward(self, x):     
        x = x.to(torch.float32)
        if self.do_bn:
            x = self.bn_en(x)

        output = x.reshape(x.shape[0], 2, -1) 
        output = self.conv1(output)
        output = self.conv2(output)
        output = output.reshape(x.shape[0], -1)
        # print(3)
        if self.do_bn:
            output = self.bn_de(output)
        out = self.decoder(output)
        
        return out
    
class CNN_LSTM(nn.Module):

    def __init__(self, features, hidden_size, class_num, num_layers = 2, dropout = 0, do_bn = False) -> None:

        super(CNN_LSTM, self).__init__()

        self.bn_en = nn.BatchNorm1d(features, momentum=0.5)
        self.conv1 = nn.Sequential(         # input shape (2, 4)
            nn.Conv1d(
                in_channels=2,              # input height
                out_channels=hidden_size,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding='same',                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (64, 4)
            nn.Dropout(dropout),
            nn.ReLU(),                      # activation
            nn.MaxPool1d(int(features / 2)),    # choose max value in 2x2 area, output shape (64, 1)
        )
        self.bn_lstm = nn.BatchNorm1d(hidden_size, momentum=0.5)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn_de = nn.BatchNorm1d(hidden_size, momentum=0.5)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, class_num), nn.Dropout(dropout))
        self.do_bn = do_bn

    def forward(self, x):       
        x = x.to(torch.float32)
        B, L, F = x.shape
        output = x.reshape(B * L, -1) 
        if self.do_bn:
            output = self.bn_en(output)

        # B, L, F = x.shape
        # output = x.reshape(B * L, -1) 
        output = output.reshape(output.shape[0], 2, -1) 
        output = self.conv1(output)

        # print(1)
        if self.do_bn:
            output = self.bn_lstm(output)   
        output = output.reshape(B, L, -1) 
        output, (cur_hidden, cur_cell) = self.lstm(output, None)
        # print(2)
        output = output.reshape(B * L, -1)
        # print(3)
        if self.do_bn:
            output = self.bn_de(output)
        output = self.decoder(output)
        output = output.reshape(B, L, -1) 
        # choose r_out at the last time step
        out = output[:, -1, :]

        return out