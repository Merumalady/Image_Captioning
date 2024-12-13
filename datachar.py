import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import re
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Define the character-level vocabulary
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 58 # Max length for the sequence of characters

# Standardization function for character-level tokenization
def custom_standardization(input_string):
    """
    Tokenize the string at the character level.
    """
    input_string = input_string.lower()  # Convert to lowercase
    # Remove any character not part of the allowed vocabulary
    input_string = re.sub(r"[^\w\s!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", "", input_string)
    return list(input_string)

class Vocabulary:
    def __init__(self, chars):
        # Initialize char2idx and idx2char from the list of characters
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}
        self.char_count = defaultdict(int)  # Count each character's occurrence
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"

    def add_sentence(self, sentence):
        """
        Add a sentence (list of characters) to the vocabulary.
        """
        for char in sentence:
            self.char_count[char] += 1
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def encode(self, sentence, max_length):
        """
        Codifica una frase en una lista de índices, agregando padding según la longitud máxima.
        """
        # Asegurarse de limitar correctamente la secuencia
        tokens = [self.sos_token] + sentence[:max_length-2] + [self.eos_token]
        token_ids = [self.char2idx.get(char, self.char2idx[self.unk_token]) for char in tokens]

        # Añadir padding si es necesario
        if len(token_ids) < max_length:
            token_ids += [self.char2idx[self.pad_token]] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]  # Limitar a max_length

        return token_ids


    def decode(self, token_ids):
        """
        Decode a list of token IDs back to the original characters.
        """
        return [self.idx2char.get(idx, self.unk_token) for idx in token_ids]

class FoodImageCaptionDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, max_seq_length=TEXT_MAX_LEN):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.max_seq_length = max_seq_length

        self.data = self.data.dropna(subset=['Title']).reset_index(drop=True)

        # Filter rows where the image doesn't exist
        self.data['Image_Path'] = self.data['Image_Name'].apply(lambda x: os.path.join(self.image_dir, f"{x}.jpg"))
        self.data = self.data[self.data['Image_Path'].apply(os.path.exists)].reset_index(drop=True)

        # Create the vocabulary
        self.vocab = Vocabulary(chars)
        self._build_vocab()

    def _build_vocab(self):
        """
        Build the vocabulary from the dataset captions (character-level tokenization).
        """
        for caption in self.data['Title']:
            tokens = custom_standardization(caption)
            self.vocab.add_sentence(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Process the image
        img_path = self.data.iloc[idx]['Image_Path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Process the caption at the character level
        caption = self.data.iloc[idx]['Title']
        tokens = custom_standardization(caption)
        encoded_caption = self.vocab.encode(tokens, self.max_seq_length)

        return image, torch.tensor(encoded_caption, dtype=torch.long)

# Custom collate function to handle padded captions
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = torch.stack(captions)  # Ya vienen paddeadas
    return images, captions


# Image transformation for data augmentation
image_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(contrast=0.3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SubsetFoodImageCaptionDataset(Dataset):
    def __init__(self, full_dataset, indices):
        self.full_dataset = full_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]

# Dataset path and directory
#csv_path = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
#image_dir = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Images\Food Images"
csv_path = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
image_dir = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Images\Food Images"

# Create the dataset
dataset = FoodImageCaptionDataset(csv_path=csv_path, image_dir=image_dir, transform=image_transforms)

# Dividir el índice de datos
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

# Crear subconjuntos para entrenamiento, validación y prueba
train_subset = SubsetFoodImageCaptionDataset(dataset, train_indices)
val_subset = SubsetFoodImageCaptionDataset(dataset, val_indices)
test_subset = SubsetFoodImageCaptionDataset(dataset, test_indices)

# Crear los dataloaders
train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, collate_fn=collate_fn)

#print(f"----> Number of training samples: {len(train_loader)}")
#print(f"----> Number of validation samples: {len(val_loader)}")
#print(f"----> Number of test samples: {len(test_loader)}")


"""
# Imprimir el vocabulario de caracteres
def print_vocab():
    print("Character vocabulary (char2idx):")
    for char, idx in char2idx.items():
        print(f"{char}: {idx}")

    print("Reverse vocabulary (idx2char):")
    for idx, char in idx2char.items():
        print(f"{idx}: {char}")

print_vocab()
"""
