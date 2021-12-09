import torch
import numpy as np
import pandas as pd
import math
import warnings
import numbers

class grud_model(torch.nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers = 1, x_mean = 0,\
     bias =True, batch_first = False, bidirectional = False, dropout_type ='mloss', dropout = 0):
        super(grud_model, self).__init__()

        self.gru_d = GRUD_cell(input_size = input_size, hidden_size= hidden_size, output_size=output_size, 
                dropout=dropout, dropout_type=dropout_type, x_mean=x_mean)
        self.hidden_to_output = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if self.num_layers >1:
            #(batch, seq, feature)
            self.gru_layers = torch.nn.GRU(
                input_size = hidden_size, hidden_size = hidden_size,batch_first = True,
                num_layers = self.num_layers -1, dropout=dropout)

    def forward(self,input):
        output, hidden = self.gru_d(input)
        # batch_size, n_hidden, n_timesteps

        if self.num_layers >1:
            output, hidden = self.gru_layers(hidden)

            output = self.hidden_to_output(output)
            output = torch.sigmoid(output)

        return output, hidden


class GRUD_cell(torch.nn.Module):
    """
    Implementation of GRUD.
    Inputs: x_mean
            n_smp x 3 x n_channels x len_seq tensor (0: data, 1: mask, 2: deltat)
    """
    def __init__(self, input_size, hidden_size, output_size=None, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0, return_hidden = False):

        print("Initializing GRU-D cell..")
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        
        super(GRUD_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if output_size:
            self.output_size = output_size
        else:
            self.output_size = hidden_size
            output_size = hidden_size

        self.num_layers = num_layers

        #controls the output, True if another GRU-D layer follows
        self.return_hidden = return_hidden 

        x_mean = torch.tensor(x_mean, dtype=torch.float32, requires_grad = True)
        self.register_buffer('x_mean', x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))
       
        #set up all the operations that are needed in the forward pass
        self.w_dg_x = torch.nn.Linear(input_size,input_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias = True)

        self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_hy = torch.nn.Linear(hidden_size, output_size, bias=True)

        Hidden_State = torch.zeros(self.hidden_size, requires_grad = True)
        #we use buffers because pytorch will take care of pushing them to GPU for us
        self.register_buffer('Hidden_State', Hidden_State)
        self.register_buffer('X_last_obs', torch.zeros(input_size)) 

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
    
    @property
    def _flat_weights(self):
        return list(self._parameters.values())


    def forward(self, input):
        X = input[:,0,:,:]
        Mask = input[:,1,:,:]
        Delta = input[:,2,:,:]
        
        output = None
        h = getattr(self, 'Hidden_State')

        #buffer system from newer pytorch version
        x_mean = getattr(self, 'x_mean')
        x_last_obsv = getattr(self, 'X_last_obs')
        

        device = next(self.parameters()).device
        output_tensor = torch.empty([X.size()[0], X.size()[2], self.output_size], dtype=X.dtype, device= device)
        hidden_tensor = torch.empty(X.size()[0], X.size()[2], self.hidden_size, dtype=X.dtype, device = device)

        #iterate over seq
        for timestep in range(X.size()[2]):
            
            x = X[:,:,timestep]
            m = Mask[:,:,timestep]
            d = Delta[:,:,timestep]

            gamma_x = torch.exp(-1* torch.nn.functional.relu( self.w_dg_x(d) ))
            gamma_h = torch.exp(-1* torch.nn.functional.relu( self.w_dg_h(d) ))

            x_last_obsv = torch.where(m>0,x,x_last_obsv)
            x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)

            if self.dropout == 0:

                h = gamma_h*h
                z = torch.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = torch.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

                h_tilde = torch.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))


                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h

                z = torch.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = torch.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))

                h_tilde = torch.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = torch.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = torch.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))
                h_tilde = torch.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''
                h = gamma_h*h

                z = torch.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = torch.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))


                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = torch.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))


                h = (1 - z) * h + z * h_tilde

            else:
                h = gamma_h * h

                z = torch.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = torch.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))
                h_tilde = torch.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde

            step_output = self.w_hy(h)
            step_output = torch.sigmoid(step_output)
            output_tensor[:,timestep,:] = step_output
            hidden_tensor[:,timestep,:] = h
        
        output = output_tensor, hidden_tensor

        return output
