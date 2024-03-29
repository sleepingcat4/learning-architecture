import torch 
from torch import nn

class LSTM(torch.nn.Module):
    def __init__(self, input_length=10, hidden_length=20):
        super(LSTM, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate components 
        self.linear_forget_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components 
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components 
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_gate = nn.Tanh()

        # output gate
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()

        self.activation_final = nn.Tanh()

        def forget(self, x, h):
            x = self.linear_forget_w1(x)
            h = self.linear_forget_r1(h)
            return self.sigmoid_forget(x + h)
        
        def input_gate(self, x, h):
            x_temp = self.linear_gate_w2(x)
            h_temp = self.linear_gate_r2(h)
            i = self.sigmoid_gate(x_temp + h_temp)
            return i 
        
        def cell_memory_gate(self, i, f, x, h, c_prev):
            x = self.linear_gate_w3(x)
            h = self.linear_gate_r3(h)

            # new information that will be injected in the new context
            k = self.activation_gate(x + h)
            g = k * i

            # forget old context/cell information
            c = f * c_prev
            # learn new context
            c_next = g + c
            return c_next
        
        def output_gate(self, x, h):
            x = self.linear_gate_w4(x)
            h = self.linear_gate_r4(h)
            return self.sigmoid_hidden_out(x + h)
        
        def forward(self, x, tuple_in):
            (h, c_prev) = tuple_in
            # Equation 1. input gate
            i = self.input_gate(x, h)
            # forget gate
            f = self.forget(x, h)
            # Euqation 3. updating the cell memory
            c_next = self.cell_memory_gate(i, f, x, h, c_prev)

            # Eq. calculate the output gate 
            o = output_gate(x, h)
            h_next = o * self.activation_final(c_next)
            return h_next, c_next