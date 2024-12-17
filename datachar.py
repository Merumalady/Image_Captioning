import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu


# Define the character-level vocabulary
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)  # 81 characters in total
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
        # Use predefined chars list for char2idx and idx2char mappings
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
        Here, we ensure that we only count characters from the predefined `chars` list.
        """
        for char in sentence:
            if char in self.char2idx:  # Only count characters that are in the predefined vocabulary
                self.char_count[char] += 1

    def encode(self, sentence, max_length):
        """
        Encode a sentence into a list of indices, adding padding according to the max length.
        """
        # Ensure the sequence is correctly truncated or padded
        tokens = [self.sos_token] + sentence[:max_length-2] + [self.eos_token]
        token_ids = [self.char2idx.get(char, self.char2idx[self.unk_token]) for char in tokens]

        # Add padding if necessary
        if len(token_ids) < max_length:
            token_ids += [self.char2idx[self.pad_token]] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]  # Limit to max_length

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

        # Create the vocabulary using the predefined chars list
        self.vocab = Vocabulary(chars)
        self._build_vocab()

    def _build_vocab(self):
        """
        Build the vocabulary from the dataset captions (character-level tokenization).
        Ensure that only characters from the predefined list are added.
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
    captions = torch.stack(captions)  # Padded captions already
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

"""
---------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------
"""

# Función para generar y guardar gráficas
def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores):
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
    plt.savefig('results/loss_accuracy_language.png')
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
    plt.savefig('results/bleu_scores_language.png')
    plt.close()

# Función de cálculo de pérdida
def compute_loss(predictions, targets, padding_idx):
    # Create the mask to ignore padding indices
    mask = (targets != padding_idx).float()
    
    # Compute the loss
    loss = torch.nn.functional.cross_entropy(predictions.reshape(-1, predictions.size(-1)), targets.reshape(-1), reduction='none')
    loss = (loss * mask.view(-1)).mean()  # Apply mask and compute mean loss
    
    return loss

# Función para calcular la exactitud (accuracy)
def compute_accuracy(predictions, targets, padding_idx=char2idx['<PAD>']):
    # Solo se calcula la exactitud para los tokens que no son padding
    pred_ids = predictions.argmax(dim=-1)
    mask = (targets != padding_idx).float()
    correct = (pred_ids[:, :-1] == targets) * mask  # Eliminar la última predicción
    accuracy = correct.sum() / mask.sum()
    return accuracy.item()

# Función para calcular BLEU-1 a BLEU-4
def compute_bleu(predictions, targets, idx2char, char2idx):
    pred_ids = predictions.argmax(dim=-1)
    pred_tokens = [idx2char[i.item()] for i in pred_ids[0]]
    target_tokens = [idx2char[i.item()] for i in targets[0] if i.item() != char2idx['<PAD>']]

    # BLEU-1 a BLEU-4
    bleu_scores = {
        "bleu_1": sentence_bleu([target_tokens], pred_tokens, weights=(1, 0, 0, 0)),
        "bleu_2": sentence_bleu([target_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0)),
        "bleu_3": sentence_bleu([target_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0)),
        "bleu_4": sentence_bleu([target_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    }
    return bleu_scores
