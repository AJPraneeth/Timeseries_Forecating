import torch
import torch.nn as nn

class LSTM_YSP(nn.Module):
    def __init__(self, input_size: int, output_size: int, device, hidden_sizes=[64, 32, 16, 8]):
        super(LSTM_YSP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.hidden_sizes = hidden_sizes

        # Dynamically create LSTMCell layers
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_sizes[i - 1], size)
            for i, size in enumerate(hidden_sizes)
        ])
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_sizes[-1], output_size)
        self.fc2 = nn.Linear(output_size, output_size)

        # Batch normalization for stability
        self.bn = nn.BatchNorm1d(hidden_sizes[-1])
        
        # Move model to the specified device
        self.to(device)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.ndim >= 2:  # Check if the parameter has at least two dimensions
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param)  # Use uniform initialization for 1D tensors
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initialize hidden and cell states for all layers
        hidden_states = [
            (torch.zeros(batch_size, size, device=x.device), torch.zeros(batch_size, size, device=x.device))
            for size in self.hidden_sizes
        ]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.lstm_cells):
                h_t, c_t = hidden_states[i]
                h_t, c_t = cell(x_t, (h_t, c_t))
                x_t = h_t  # Pass the hidden state to the next layer
                hidden_states[i] = (h_t, c_t)

        # Apply batch normalization
        h_t_final = self.bn(hidden_states[-1][0])

        # Predictions for multiple days
        out1 = self.fc1(h_t_final)  # First prediction
        out2 = self.fc2(out1)      # Second prediction

        # Stack predictions along a new dimension
        final_out = torch.stack([out1, out2], dim=1)  # Shape: (batch_size, 2, output_size)
        return final_out
