import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
from nltk.translate.bleu_score import sentence_bleu
import evaluate

# Load evaluation metrics
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

# Positional Embedding for Transformer
class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_embedding(tgt_embedded)

        for layer in self.layers:
            tgt_embedded = layer(tgt_embedded, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)

        logits = self.fc_out(self.norm(tgt_embedded))
        return self.softmax(logits)

# Transformer-based Character-level Model
class TransformerCharacterLevel(nn.Module):
    def __init__(self, embed_dim, num_heads, dense_dim, vocab_size, num_layers=2, max_seq_len=50):
        super(TransformerCharacterLevel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(sequence_length=max_seq_len, embed_dim=embed_dim)
        self.encoder = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, dense_dim=dense_dim, num_layers=num_layers)
        self.decoder = TransformerDecoder(embed_dim=embed_dim, num_heads=num_heads, dense_dim=dense_dim, vocab_size=vocab_size, num_layers=num_layers, max_seq_len=max_seq_len)

    def forward(self, inputs, targets, tgt_mask=None, memory_mask=None):
        input_embedded = self.embedding(inputs)
        input_embedded = self.positional_embedding(input_embedded)

        memory = self.encoder(input_embedded)
        outputs = self.decoder(targets, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return outputs

    def generate(self, inputs, max_len=20, start_idx=2, end_idx=3):
        input_embedded = self.embedding(inputs)
        input_embedded = self.positional_embedding(input_embedded)

        memory = self.encoder(input_embedded)

        outputs = []
        tgt = torch.tensor([[start_idx]], device=inputs.device)

        for _ in range(max_len):
            preds = self.decoder(tgt, memory)
            next_char = preds[:, -1, :].argmax(dim=-1).item()
            outputs.append(next_char)
            if next_char == end_idx:
                break
            tgt = torch.cat([tgt, torch.tensor([[next_char]], device=inputs.device)], dim=1)

        return outputs

# Evaluation and Training Functions

def evaluate_model(model, data_loader, vocab):
    model.eval()
    total_bleu1 = total_bleu2 = total_rouge = total_meteor = 0
    count = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = [model.generate(input.unsqueeze(0), max_len=20, start_idx=vocab.word2idx["<START>"], end_idx=vocab.word2idx["<END>"])
                       for input in inputs]
            predicted = [vocab.decode(output) for output in outputs]

            predicted = [
                " ".join([char for char in caption if char not in ["<START>", "<END>", "<PAD>"]])
                for caption in predicted
            ]

            references = [vocab.decode(target.tolist()) for target in targets]
            references = [
                " ".join([char for char in ref if char not in ["<START>", "<END>", "<PAD>"]])
                for ref in references
            ]

            bleu1 = bleu.compute(predictions=predicted, references=references, max_order=1)
            bleu2 = bleu.compute(predictions=predicted, references=references, max_order=2)
            total_bleu1 += bleu1["bleu"]
            total_bleu2 += bleu2["bleu"]

            res_r = rouge.compute(predictions=predicted, references=references)
            total_rouge += res_r['rougeL']

            res_m = meteor.compute(predictions=predicted, references=references)
            total_meteor += res_m['meteor']

            count += 1

    avg_bleu1 = total_bleu1 / count
    avg_bleu2 = total_bleu2 / count
    avg_rouge = total_rouge / count
    avg_meteor = total_meteor / count

    print(f"Average BLEU1 Score: {avg_bleu1*100:.4f}%")
    print(f"Average BLEU2 Score: {avg_bleu2*100:.4f}%")
    print(f"Average ROUGE-L Score: {avg_rouge*100:.4f}%")
    print(f"Average METEOR Score: {avg_meteor*100:.4f}%")

def train_model(model, name, train_loader, val_loader, vocab, num_epochs=20, learning_rate=1e-3, optimizer_type="adam"):
    criterion = CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD
    }
    optimizer = optimizers[optimizer_type](model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, targets)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        evaluate_model(model, val_loader, vocab)

        save_model(
            model=model,
            optimizer=optimizer,
            vocab=vocab,
            loss=avg_loss,
            name=name,
            save_dir=r"C:/Users/merit/OneDrive/Escritorio/Image_Captioning/"
        )

def save_model(model, optimizer, vocab, loss, name, save_dir="./checkpoints/"):
    """
    Saves the trained model, optimizer state, and vocabulary as a whole.

    Args:
        model (nn.Module): The trained model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        vocab (object): The vocabulary object to save.
        loss (float): The final loss value.
        name (str): A base name for the saved files.
        save_dir (str): Directory where the model and vocab will be saved. Defaults to './checkpoints/'.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the entire model checkpoint (model + optimizer state)
    model_filename = os.path.join(save_dir, f"{name}_model.pth")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "vocab": vocab
    }
    torch.save(checkpoint, model_filename)
    print(f"Model and optimizer checkpoint saved at {model_filename}")

if __name__ == "__main__":
    import data as d  # Assuming data.py contains your dataset and vocab loaders

    # Define device for model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    embed_size = 256
    num_heads = 8
    dense_dim = 512
    num_layers = 6
    max_seq_len = 50
    vocab_size = len(d.dataset.vocab.word2idx)

    # Initialize the model
    model = TransformerCharacterLevel(
        embed_dim=embed_size,
        num_heads=num_heads,
        dense_dim=dense_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    ).to(device)

    # Train the model
    train_model(
        model=model,
        name="transformer_captioning",
        train_loader=d.train_loader,
        val_loader=d.val_loader,
        vocab=d.dataset.vocab,
        num_epochs=15,
        learning_rate=1e-4,
        optimizer_type="adam"
    )
