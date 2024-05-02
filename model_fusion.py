import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Generate a MLP cell
    """
    def __init__(self, features, hidden_size, class_num, dropout = 0, do_bn = False):
        
        super(MLP, self).__init__()
        
        self.bn_en = nn.BatchNorm1d(features, momentum=0.5)
        self.mlp_sub = nn.Sequential(
            nn.Linear(features, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU()
            )
        self.bn_de = nn.BatchNorm1d(hidden_size, momentum=0.5)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, class_num), nn.Dropout(dropout))
        self.do_bn = do_bn
        
    def forward(self, x):  
        x = x.to(torch.float32)         
        if self.do_bn:
            output = self.bn_en(x)
        else:
            output = x
          
        output = self.mlp_sub(output)
     
        if self.do_bn:
            output = self.bn_de(output)
        out = self.decoder(output)
        
        return out
    
class MLP_fusion(nn.Module):

    def __init__(self, features, features_mm, wb_sel, hidden_size, class_num, dropout = 0, do_bn = False) -> None:

        super(MLP_fusion, self).__init__()

        self.sub = MLP(features, hidden_size[0], class_num, dropout, do_bn)
        self.mm = MLP(features_mm, hidden_size[1], class_num, dropout, do_bn)
        self.bn_att = nn.BatchNorm1d(class_num*2, momentum=0.5)
        self.att = nn.Sequential(nn.Linear(class_num*2, class_num*2), nn.Dropout(dropout), nn.Sigmoid())
        self.fusion = MLP(class_num*2, hidden_size[2], class_num, dropout, do_bn)
        self.do_bn = do_bn
        self.wb_sel = wb_sel
        self.wb_num = int(class_num / features_mm * 2)

    def forward(self, x, y):        
        x = x.to(torch.float32)  
        y = y.to(torch.float32)
        out_sub = self.sub(x)        
        p_mm = torch.zeros(y.shape[0],int(y.shape[1]/2))
        out_sub_re = torch.reshape(out_sub,(out_sub.shape[0],int(out_sub.shape[1]/self.wb_num),self.wb_num))
        p_mm = torch.sum(out_sub_re, 2)

        index_mm = torch.topk(p_mm, self.wb_sel, 1)[1].data.cpu().numpy()

        input_mm = torch.zeros(y.shape, device=torch.device('cuda'))        
        for i in range(y.shape[0]):
            input_mm[i, index_mm[i]*2] = y[i, index_mm[i]*2]
            input_mm[i, index_mm[i]*2+1] = y[i, index_mm[i]*2+1]
        out_mm = self.mm(input_mm)
        if self.do_bn:
            output = self.bn_att(torch.cat((out_sub, out_mm), 1))
        else:
            output = torch.cat((out_sub, out_mm), 1)
        out_att = self.att(output)
        output = torch.mul(output, out_att)
        output = self.fusion(output)

        return out_sub, out_mm, output

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
        x = x.to(torch.float32)
        B, L, F = x.shape
        output = x.reshape(B * L, -1) 
        if self.do_bn:
            output = self.bn_en(output)
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
      
    
class LSTM_fusion(nn.Module):

    def __init__(self, features, features_mm, wb_sel, hidden_size, class_num, num_layers = 2, dropout = 0, do_bn = False) -> None:

        super(LSTM_fusion, self).__init__()

        self.sub = LSTM(features, hidden_size[0], class_num, num_layers, dropout, do_bn)
        self.mm = MLP(features_mm, hidden_size[1], class_num, dropout, do_bn)
        self.bn_att = nn.BatchNorm1d(class_num*2, momentum=0.5)
        self.att = nn.Sequential(nn.Linear(class_num*2, class_num*2), nn.Dropout(dropout), nn.Sigmoid())
        self.fusion = MLP(class_num*2, hidden_size[2], class_num, dropout, do_bn)
        self.do_bn = do_bn
        self.wb_sel = wb_sel
        self.wb_num = int(class_num / features_mm * 2)

    def forward(self, x, y):      
        y = y.to(torch.float32)    
        out_sub = self.sub(x)  
        p_mm = torch.zeros(y.shape[0],int(y.shape[1]/2))
        out_sub_re = torch.reshape(out_sub,(out_sub.shape[0],int(out_sub.shape[1]/self.wb_num),self.wb_num))
        p_mm = torch.sum(out_sub_re, 2)

        index_mm = torch.topk(p_mm, self.wb_sel, 1)[1].data.cpu().numpy()
        input_mm = torch.zeros(y.shape, device=torch.device('cuda'))
        for i in range(y.shape[0]):
            input_mm[i, index_mm[i]*2] = y[i, index_mm[i]*2]
            input_mm[i, index_mm[i]*2+1] = y[i, index_mm[i]*2+1]
        out_mm = self.mm(input_mm)

        if self.do_bn:
            output = self.bn_att(torch.cat((out_sub, out_mm), 1))
        else:
            output = torch.cat((out_sub, out_mm), 1)  
        out_att = self.att(output)
        output = torch.mul(output, out_att)       
        output = self.fusion(output)

        return out_sub, out_mm, output


class CNN(nn.Module):
    """
    Generate a CNN cell
    """
    def __init__(self, features, in_channels, hidden_size, class_num, dropout = 0, do_bn = False):
        
        super(CNN, self).__init__()
        
        self.bn_en = nn.BatchNorm1d(features, momentum=0.5)
        self.conv1 = nn.Sequential(         
            nn.Conv1d(
                in_channels=in_channels,              # input height
                out_channels=hidden_size[0],            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding='same',                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              
            nn.Dropout(dropout),
            nn.ReLU(),                      # activation
        )
        self.conv2 = nn.Sequential(         
            nn.Conv1d(hidden_size[0], hidden_size[1], 3, 1, 'same'),     
            nn.Dropout(dropout),
            nn.ReLU(),                      # activation
            nn.MaxPool1d(int(features / in_channels)),                
        )
        self.bn_de = nn.BatchNorm1d(hidden_size[1], momentum=0.5)
        self.decoder = nn.Sequential(nn.Linear(hidden_size[1], class_num), nn.Dropout(dropout))
        self.do_bn = do_bn
        self.in_channels = in_channels
        
    def forward(self, x):     
        if self.do_bn:
            output = self.bn_en(x)
        else:
            output = x

        output = output.reshape(x.shape[0], self.in_channels, -1) 
        output = self.conv1(output)
        output = self.conv2(output)
        output = output.reshape(x.shape[0], -1)
        # print(3)
        if self.do_bn:
            output = self.bn_de(output)
        out = self.decoder(output)
        
        return out    
    
class CNN_fusion(nn.Module):

    def __init__(self, features, features_mm, wb_sel, hidden_size, class_num, num_layers = 2, dropout = 0, do_bn = False) -> None:

        super(CNN_fusion, self).__init__()

        self.sub = CNN(features, 2, hidden_size, class_num, dropout, do_bn)
        self.mm = CNN(features_mm, 2, hidden_size, class_num, dropout, do_bn)
        self.bn_att = nn.BatchNorm1d(class_num*2, momentum=0.5)
        self.att = nn.Sequential(nn.Linear(class_num*2, class_num*2), nn.Dropout(dropout), nn.Sigmoid())
        self.fusion = CNN(class_num*2, 1, hidden_size, class_num, dropout, do_bn)
        self.do_bn = do_bn
        self.wb_sel = wb_sel
        self.wb_num = int(class_num / features_mm * 2)

    def forward(self, x, y):       
        x = x.to(torch.float32)  
        y = y.to(torch.float32)   
        out_sub = self.sub(x)        
        p_mm = torch.zeros(y.shape[0],int(y.shape[1]/2))
        out_sub_re = torch.reshape(out_sub,(out_sub.shape[0],int(out_sub.shape[1]/self.wb_num),self.wb_num))
        p_mm = torch.sum(out_sub_re, 2)

        index_mm = torch.topk(p_mm, self.wb_sel, 1)[1].data.cpu().numpy()
        input_mm = torch.zeros(y.shape, device=torch.device('cuda'))
        for i in range(y.shape[0]):
            input_mm[i, index_mm[i]*2] = y[i, index_mm[i]*2]
            input_mm[i, index_mm[i]*2+1] = y[i, index_mm[i]*2+1]
        out_mm = self.mm(input_mm)

        if self.do_bn:
            output = self.bn_att(torch.cat((out_sub, out_mm), 1))
        else:
            output = torch.cat((out_sub, out_mm), 1)
        
        out_att = self.att(output)
        output = torch.mul(output, out_att)
        output = self.fusion(output)

        return out_sub, out_mm, output