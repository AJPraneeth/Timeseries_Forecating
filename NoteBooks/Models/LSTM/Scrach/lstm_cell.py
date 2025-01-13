import torch 
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,input_size:torch.Tensor,hiden_size:torch.Tensor, *args, **kwargs):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.hiden_size = hiden_size
        
        #Input gate Weight
        self.W_ii = nn.Parameter(torch.Tensor(input_size,hiden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hiden_size,hiden_size))
        self.b_i = nn.Parameter(torch.Tensor(hiden_size))
        
        #Forget gate Weight
        self.W_if = nn.Parameter(torch.Tensor(input_size,hiden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hiden_size,hiden_size))
        self.b_f = nn.Parameter(torch.Tensor(hiden_size))
        
        #Output gate Weight
        self.W_io = nn.Parameter(torch.Tensor(input_size,hiden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hiden_size,hiden_size))
        self.b_o = nn.Parameter(torch.Tensor(hiden_size))
        
        #Cell gate Weight
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hiden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hiden_size,hiden_size))
        self.b_g = nn.Parameter(torch.Tensor(hiden_size))
        
        self.init_weights()
        
    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
            
    def forward(self,x,h_prev,c_prev):
        i_t= torch.sigmoid(torch.mm(x,self.W_ii)+torch.mm(h_prev,self.W_hi)+self.b_i)
        f_t= torch.sigmoid(torch.mm(x,self.W_if)+torch.mm(h_prev,self.W_hf)+self.b_f)
        o_t= torch.sigmoid(torch.mm(x,self.W_io)+torch.mm(h_prev,self.W_ho)+self.b_o)
        g_t= torch.tanh(torch.mm(x,self.W_ig)+torch.mm(h_prev,self.W_hg)+self.b_g)
        
        c_t = f_t*c_prev + i_t*g_t
        h_t = o_t*torch.tanh(c_t)
        return h_t,c_t
        
            
        
        
        