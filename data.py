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


import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

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
        # Procesar imagen
        img_path = self.data.iloc[idx]['Image_Path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        

        # Procesar texto
        caption = self.data.iloc[idx]['Title']
        return image, caption

#image_dir = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Images\Food Images"
#dataset = FoodImageCaptionDataset(csv_path=r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv", image_dir=image_dir)
#image_dir = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Images\Food Images"
#dataset = FoodImageCaptionDataset(csv_path=r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv", image_dir=image_dir)

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

nltk.download('punkt')
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
    
    # (Opcional) Procesar con spaCy: Lematización y eliminación de stopwords
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    
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
image_dir = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Images\Food Images"

dataset = FoodImageCaptionDataset(csv_path=csv_path, image_dir=image_dir, transform=image_transforms)

# Dividir datos en entrenamiento, validación y pruebas
train_data, val_data, test_data = train_val_split(dataset, validation_size=0.2, test_size=0.02)

# Crear dataloaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

# Spliting the dataset
print(f"Total number of samples: {len(dataset)}")
print(f"----> Number of training samples: {len(train_data)}")
print(f"----> Number of validation samples: {len(val_data)}")
print(f"----> Number of test samples: {len(test_data)}")
