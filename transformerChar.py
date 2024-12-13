import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
#from nltk.translate.bleu_score import sentence_bleu
import evaluate

# Load evaluation metrics
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

# Positional Embedding for Transformer
class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.positional_embeddings = nn.Parameter(torch.zeros(sequence_length, embed_dim))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_embeddings[:seq_len, :].unsqueeze(0)

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dense_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=dense_dim, dropout=0.1, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dense_dim, vocab_size, num_layers, max_seq_len=50):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(sequence_length=max_seq_len, embed_dim=embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=dense_dim, dropout=0.1, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs, targets, tgt_mask=None, memory_mask=None):
        # Asegurarse de que los índices sean de tipo LongTensor
        inputs = inputs.long()
        targets = targets.long()

        input_embedded = self.embedding(inputs)
        input_embedded = self.positional_embedding(input_embedded)
        memory = self.encoder(input_embedded)
        outputs = self.decoder(targets, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return outputs


# Transformer Model
class TransformerCharacterLevel(nn.Module):
    def __init__(self, embed_dim, num_heads, dense_dim, vocab_size, num_layers=2, max_seq_len=50):
        super(TransformerCharacterLevel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(sequence_length=max_seq_len, embed_dim=embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, dense_dim, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, dense_dim, vocab_size, num_layers, max_seq_len)

    def forward(self, inputs, targets, tgt_mask=None, memory_mask=None):
        input_embedded = self.embedding(inputs)
        input_embedded = self.positional_embedding(input_embedded)
        memory = self.encoder(input_embedded)
        outputs = self.decoder(targets, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return outputs

# Training and Evaluation Functions
def train_model(model, name, train_loader, val_loader, vocab, num_epochs=20, learning_rate=1e-3):
    criterion = CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, targets[:, :-1])
            loss = criterion(outputs.view(-1, len(vocab.char2idx)), targets[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss = evaluate_model(model, val_loader, vocab, criterion)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        save_model(model, r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\{name}_epoch_{epoch+1}.pth")

    plot_training(train_losses, val_losses, r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\{name}_training_plot.png")

def evaluate_model(model, data_loader, vocab, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets[:, :-1])
            loss = criterion(outputs.view(-1, len(vocab.char2idx)), targets[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def plot_training(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")

if __name__ == "__main__":
    import datachar as d
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size = 256
    num_heads = 8
    dense_dim = 512
    num_layers = 6
    max_seq_len = 50
    vocab_size = len(d.train_dataset.vocab.char2idx)

    model = TransformerCharacterLevel(
        embed_dim=embed_size, num_heads=num_heads, dense_dim=dense_dim, vocab_size=vocab_size, num_layers=num_layers, max_seq_len=max_seq_len
    ).to(device)

    train_model(model, "transformer_captioning", d.train_loader, d.val_loader, d.train_dataset.vocab, num_epochs=15, learning_rate=1e-4)
