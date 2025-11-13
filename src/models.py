import torch
import torch.nn as nn

def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "tanh":
        return nn.Tanh()
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation: {name}")


# -------------------------------------------------
# Simple RNN
# -------------------------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, activation="relu", embedding_dim=100, hidden_size=64, num_layers=2, dropout=0.5):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.out_act = nn.Sigmoid()
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])  # last timestep
        out = self.act(out)
        out = self.fc(out)
        out = self.out_act(out)
        return out


# -------------------------------------------------
# LSTM
# -------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, activation="relu", embedding_dim=100, hidden_size=64, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.out_act = nn.Sigmoid()
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.act(out)
        out = self.fc(out)
        out = self.out_act(out)
        return out


# -------------------------------------------------
# Bidirectional LSTM
# -------------------------------------------------
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, activation="relu", embedding_dim=100, hidden_size=64, num_layers=2, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1) 
        self.out_act = nn.Sigmoid()
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.bilstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.act(out)
        out = self.fc(out)
        out = self.out_act(out)
        return out


# -------------------------------------------------
# Model Summary Utility
# -------------------------------------------------
def describe_model(model):
    """
    Prints a clean summary of the model architecture and key parameters.
    Works for RNN, LSTM, and BiLSTM classes defined in this module.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print(f"Model Architecture: {model.__class__.__name__}")
    print("-" * 60)
    print(model)
    print("-" * 60)

    # Basic statistics
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # Embedding details
    if hasattr(model, 'embedding'):
        print(f"Embedding Dimension  : {model.embedding.embedding_dim}")

    # Identify recurrent type
    if hasattr(model, 'rnn'):
        rnn_type = "Simple RNN"
    elif hasattr(model, 'lstm') and not hasattr(model, 'bilstm'):
        rnn_type = "LSTM"
    elif hasattr(model, 'bilstm'):
        rnn_type = "Bidirectional LSTM"
    else:
        rnn_type = "Unknown"
    print(f"Recurrent Type       : {rnn_type}")

    # Common hyperparameters
    print(f"Hidden Size          : 64")
    print(f"Number of Layers     : 2")
    print(f"Dropout Rate         : 0.5")

    # Activation details
    activation = model.act.__class__.__name__ if hasattr(model, 'act') else "N/A"
    print(f"Activation Function  : {activation}")
    print(f"Output Activation    : Sigmoid")
    print("=" * 60)

if __name__ == "__main__":
    vocab_size = 10000

    print("\nRunning model summaries...\n")

    # Instantiate each model
    rnn_model = SimpleRNNModel(vocab_size=vocab_size, activation="relu")
    lstm_model = LSTMModel(vocab_size=vocab_size, activation="relu")
    bilstm_model = BiLSTMModel(vocab_size=vocab_size, activation="relu")

    # Print summaries
    describe_model(rnn_model)
    describe_model(lstm_model)
    describe_model(bilstm_model)

    print("\nModel summaries generated successfully!\n")

