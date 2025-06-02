from torch import nn
import torch

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.classifier(hn[-1]) # Lasti hidden layer
        return out

class ConvLSTM(nn.Module):
    def __init__(self, input_size, extracted_features, kernel_size, padding, hidden_size, num_layers, num_classes):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=extracted_features, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(extracted_features)
        )
        self.lstm = nn.LSTM(extracted_features, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # → (batch, input_size, seq_len)
        x = self.conv(x) # requires (batch, channels, seq_len)
        x = x.transpose(1, 2)  # → (batch, seq_len, conv_channels)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.classifier(hn[-1])
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
    
    def forward(self, lstm_outputs, lengths):
        attention_scores = self.attention(lstm_outputs).squeeze(-1)  # (batch, seq_len)

        len = lstm_outputs.size(1)
        mask = torch.arange(len, device=lstm_outputs.device)[None, :] < lengths[:, None]
        '''
        mask out the padded values in the attention scores.
        for len = [3, 2]
        mask = [[True, True, True],
                [True, True, False]]
        '''
        attention_scores[~mask] = float('-inf')  # Set scores for padded positions to -inf
        
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, seq_len, 1)
        
        context_vector = torch.sum(lstm_outputs * attention_weights, dim=1)  # (batch, hidden_size)
        return context_vector, attention_weights

class AttentionLSTM(nn.Module):
    def __init__ (self, input_size, hidden_size, num_layers, num_classes):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x, lengths): 
        sorted_lens, indices = lengths.sort(descending=True)
        sorted_x = x[indices]
        packed = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_lens.cpu(), batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        _,unsorted_indices = indices.sort()
        lstm_out = lstm_out[unsorted_indices]  # Unsort the output to match original order
        lengths = lengths[unsorted_indices]  # Unsort the lengths to match original order

        context_vector, weights = self.attention(lstm_out, lengths)
        classes = self.classifier(context_vector)
        return classes

class NextLocationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_size, 3)  # Output: (latitude, longitude, timestamp)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _ , (hn, _) = self.lstm(packed)

        output = self.regressor(hn[-1])
        return output


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_out, c_out) = self.lstm(packed)
        return h_out, c_out

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        output, (h_out, c_out) = self.lstm(x, (h, c))
        pred = self.regressor(output)
        return pred, h_out, c_out

class NextSequenceLSTM(nn.Module):
    def __init__(self, input_size = 3, output_size = 3,
                 hidden_size_encoder = 64,
                 num_layers_encoder = 1, 
                 hidden_size_decoder = 64,
                 num_layers_decoder = 1,
                 pred_length = 5,
                 teacher_forcing_ratio = 0.5):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size_encoder, num_layers_encoder)
        self.decoder = Decoder(output_size, hidden_size_decoder, num_layers_decoder)
        self.pred_length = pred_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, source, source_lengths, target=None):
        hidden, cell = self.encoder(source, source_lengths)

        # First decoder input (start with last known point)
        decoder_input = source[:, -1, :3].unsqueeze(1)  # only (lat, lon, timestamp)

        outputs = []

        for t in range(self.pred_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)

            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                # Teacher forcing: feed true next target
                decoder_input = target[:, t].unsqueeze(1)  # (batch, 1, 3)
            else:
                # No teacher forcing: feed prediction
                decoder_input = output

        outputs = torch.cat(outputs, dim=1)  # (batch, pred_length, 3)
        return outputs