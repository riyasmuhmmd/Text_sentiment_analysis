
import torch
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input to (batch_size, sequence_length, input_size)
        # Assuming sequence_length is 1 in this case
        x = x.unsqueeze(1)  # Adding a dimension for sequence length

        # Check if input is batched or not
        is_batched = x.dim() == 3

        # Initialize hidden and cell states with correct dimensions for batch_first=True
        if is_batched:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        else:
            # For unbatched input, create 2-D hidden and cell states
            h0 = torch.zeros(self.layer_dim, self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.layer_dim, self.hidden_dim).requires_grad_()

        # Detaching h0 and c0 from the computation graph is optional
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Now 'out' has dimensions (batch_size, sequence_length, hidden_size)
        # Select the output from the last time step
        out = self.fc(out[:, -1, :])
        return out
