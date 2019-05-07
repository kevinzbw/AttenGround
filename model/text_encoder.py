import torch
import torch.nn as nn

class TextEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=False)
        self.device = device
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device), torch.randn(2, 1, self.hidden_dim // 2).to(self.device))
    
    def get_hidden(self):
        return self.hidden

    def _get_word_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        word_features = lstm_out.transpose(0, 1)
        return word_features

    def forward(self, sentence):
        word_features = self._get_word_features(sentence)
        return word_features