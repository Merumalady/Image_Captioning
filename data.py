from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from torchvision import transforms
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
import spacy
import re
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

class FoodImageCaptionDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, max_seq_length=10):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.max_seq_length = max_seq_length

        self.data = self.data.dropna(subset=['Title']).reset_index(drop=True)

        # Filtrar filas donde la imagen no existe
        self.data['Image_Path'] = self.data['Image_Name'].apply(lambda x: os.path.join(self.image_dir, f"{x}.jpg"))
        self.data = self.data[self.data['Image_Path'].apply(os.path.exists)].reset_index(drop=True)
        
        # Crear vocabulario
        self.vocab = Vocabulary()
        self._build_vocab()

    def _build_vocab(self):
        # Añadir todas las descripciones al vocabulario
        for caption in self.data['Title']:
            tokens = custom_standardization(caption)
            self.vocab.add_sentence(tokens)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(f"Processing item {idx}/{len(self.data)}: {self.image_dir}")
        # Procesar imagen
        img_path = self.data.iloc[idx]['Image_Path']
        print(f"Processing item {idx}/{len(self.data)}: {img_path}")  # Place the print after initializing img_path

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Procesar texto
        caption = self.data.iloc[idx]['Title']
        tokens = custom_standardization(caption)
        encoded_caption = self.vocab.encode(tokens, self.max_seq_length)

        return image, torch.tensor(encoded_caption, dtype=torch.long)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = [torch.tensor(caption, dtype=torch.long) for caption in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions



def train_val_split(data, validation_size=0.2, test_size=0.02):
    """
    Splits the data into training, validation, and test sets.

    Args:
        data (pd.DataFrame): The input DataFrame containing the dataset.
        validation_size (float): Proportion of the data to use for validation.
        test_size (float): Proportion of the validation data to use for testing.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        tuple: (train_data, validation_data, test_data) as DataFrames.
    """
    # Split the dataset
    train_data, validation_data = train_test_split(data, test_size=validation_size, random_state=42)
    validation_data, test_data = train_test_split(validation_data, test_size=test_size, random_state=42)

    return train_data, validation_data, test_data

nltk.download('punkt', quiet= True)
nlp = spacy.load("en_core_web_sm")

def custom_standardization(input_string):
    """
    Normaliza el texto:
    - Convierte a minúsculas
    - Elimina caracteres no deseados
    - Tokeniza el texto
    - (Opcional) Lematiza y elimina stopwords con spaCy
    """
    # Convertir a minúsculas
    input_string = input_string.lower()
    
    # Eliminar caracteres no deseados
    input_string = re.sub(r"[^\w\s]", "", input_string)  # Elimina puntuación
    input_string = re.sub(r"\d+", "", input_string)  # Elimina números
    
    # Tokenizar con NLTK
    tokens = word_tokenize(input_string)
    return tokens

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = defaultdict(int)
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.build_vocab()
    
    def build_vocab(self):
        # Inicializar con tokens especiales
        self.word2idx = {self.pad_token: 0, self.unk_token: 1, self.start_token: 2, self.end_token: 3}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def add_sentence(self, sentence):
        for word in sentence:
            self.word_count[word] += 1
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, sentence, max_length):
        # Convertir palabras a índices
        tokens = [self.start_token] + sentence + [self.end_token]
        token_ids = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in tokens]
        # Padding o truncamiento
        token_ids = token_ids[:max_length] + [self.word2idx[self.pad_token]] * (max_length - len(token_ids))
        return token_ids
    
    def decode(self, token_ids):
        # Convertir índices a palabras
        return [self.idx2word.get(idx, self.unk_token) for idx in token_ids]


class SubsetFoodImageCaptionDataset(Dataset):
    def __init__(self, full_dataset, indices):
        self.full_dataset = full_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]

# Transformaciones de imágenes
image_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(contrast=0.3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Crear el dataset
csv_path = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
image_dir = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Images/Food Images"


"""
---------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------
"""

# Función para generar y guardar gráficas
def plot_metrics(name_model, train_losses, train_accuracies, val_losses, val_accuracies, bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores):
    # Gráfico de pérdidas y exactitud
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/loss_accuracy_{name_model}.png')
    plt.close()

    # Gráficos de BLEU
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.plot(bleu_1_scores, label='BLEU-1')
    plt.title('BLEU-1')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-1')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(bleu_2_scores, label='BLEU-2')
    plt.title('BLEU-2')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-2')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(bleu_3_scores, label='BLEU-3')
    plt.title('BLEU-3')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-3')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(bleu_4_scores, label='BLEU-4')
    plt.title('BLEU-4')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-4')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/bleu_scores_{name_model}.png')
    plt.close()


# Función de cálculo de pérdida
def compute_loss(predictions, targets, padding_idx):
    # Create the mask to ignore padding indices
    mask = (targets != padding_idx).float()
    
    # Compute the loss
    loss = torch.nn.functional.cross_entropy(predictions.reshape(-1, predictions.size(-1)), targets.reshape(-1), reduction='none')
    loss = (loss * mask.view(-1)).mean()  # Apply mask and compute mean loss
    
    return loss


def compute_accuracy(predictions, targets, padding_idx):
    # Solo se calcula la exactitud para los tokens que no son padding
    pred_ids = predictions.argmax(dim=-1)
    mask = (targets != padding_idx).float()
    correct = (pred_ids[:, :-1] == targets) * mask  # Eliminar la última predicción
    accuracy = correct.sum() / mask.sum()
    return accuracy.item()
    
# Función para calcular BLEU-1 a BLEU-4
def compute_bleu(predictions, targets, idx2word, word2idx):
    pred_ids = predictions.argmax(dim=-1)
    pred_tokens = [idx2word[i.item()] for i in pred_ids[0]]
    target_tokens = [idx2word[i.item()] for i in targets[0] if i.item() != word2idx['<PAD>']]

    # BLEU-1 a BLEU-4
    bleu_scores = {
        "bleu_1": sentence_bleu([target_tokens], pred_tokens, weights=(1, 0, 0, 0)),
        "bleu_2": sentence_bleu([target_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0)),
        "bleu_3": sentence_bleu([target_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0)),
        "bleu_4": sentence_bleu([target_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    }
    return bleu_scores
