import torch
import torch.nn as nn

class TextDecoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, device):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, dropout=dropout)
        
        self.hidden2wordset = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input)).view(1, 1, -1)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = lstm_out.view(1, self.hidden_dim)
        prediction = self.hidden2wordset(lstm_out)
        return prediction, hidden
       
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device), torch.randn(2, 1, self.hidden_dim // 2).to(self.device))