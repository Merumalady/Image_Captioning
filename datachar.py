import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import re

# Define the character-level vocabulary
chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201  # Max length for the sequence of characters

class FoodImageCaptionDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, max_seq_length=TEXT_MAX_LEN):
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
        # Añadir todas las descripciones al vocabulario a nivel de caracteres
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
        
        # Procesar texto a nivel de caracteres
        caption = self.data.iloc[idx]['Title']
        tokens = custom_standardization(caption)
        encoded_caption = self.vocab.encode(tokens, self.max_seq_length)

        return image, torch.tensor(encoded_caption, dtype=torch.long)

# Custom collate function to handle padded captions
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = [torch.tensor(caption, dtype=torch.long) for caption in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=char2idx['<PAD>'])
    return images, captions

# Function to split the dataset
def train_val_split(data, validation_size=0.2, test_size=0.02):
    train_data, validation_data = train_test_split(data, test_size=validation_size, random_state=42)
    validation_data, test_data = train_test_split(validation_data, test_size=test_size, random_state=42)
    return train_data, validation_data, test_data

# Tokenizer at the character level
def custom_standardization(input_string):
    """
    Tokenize the string at the character level.
    """
    input_string = input_string.lower()
    # Remove characters not part of our vocabulary
    input_string = re.sub(r"[^\w\s!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", "", input_string)
    return list(input_string)

class Vocabulary:
    def __init__(self):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.char_count = defaultdict(int)
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"

    def add_sentence(self, sentence):
        for char in sentence:
            self.char_count[char] += 1
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def encode(self, sentence, max_length):
        # Add <SOS> and <EOS> tokens
        tokens = [self.sos_token] + sentence + [self.eos_token]
        token_ids = [self.char2idx.get(char, self.char2idx[self.unk_token]) for char in tokens]
        # Padding or truncating
        token_ids = token_ids[:max_length] + [self.char2idx[self.pad_token]] * (max_length - len(token_ids))
        return token_ids

    def decode(self, token_ids):
        return [self.idx2char.get(idx, self.unk_token) for idx in token_ids]

# Image transformation for data augmentation
image_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(contrast=0.3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset path and directory
csv_path = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
image_dir = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Images\Food Images"
#csv_path = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
#image_dir = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Images\Food Images"

# Create the dataset
dataset = FoodImageCaptionDataset(csv_path=csv_path, image_dir=image_dir, transform=image_transforms)

# Split data into training, validation, and test sets
train_data, val_data, test_data = train_val_split(dataset.data, validation_size=0.1, test_size=0.1)

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)
