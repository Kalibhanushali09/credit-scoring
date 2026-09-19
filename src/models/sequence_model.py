import torch
import numpy as np
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,   
            hidden_size=hidden_size, 
            batch_first=True       
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        embedding = h_n.squeeze(0)
        return embedding

def generate_embeddings(sequences, hidden_size=64, batch_size=512):
    if sequences is None or len(sequences) == 0:
        return np.empty((0, hidden_size))

    model = LSTMModel(input_size=3, hidden_size=hidden_size)
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            tensor = torch.from_numpy(batch).float()
            embeddings = model(tensor)
            all_embeddings.append(embeddings.detach().numpy())

    return np.concatenate(all_embeddings, axis=0)