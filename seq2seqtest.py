import torch
import pickle
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
import data as d
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader
from torchvision import transforms
import evaluate


bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

# Definir el dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para cargar el vocabulario
def load_vocab(filename="vocab.pkl"):
    with open(filename, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {filename}")
    return vocab

# Modelo Encoder-Decoder (como el tuyo, puedes personalizarlo)
class EncoderDecoderModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(EncoderDecoderModel, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        
        self.decoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions):
        features = self.resnet(images)
        output, hidden = self.decoder(features.unsqueeze(1))
        out = self.fc_out(output.squeeze(1))
        return out

    def generate_caption(self, image, vocab, max_len=20):
        features = self.resnet(image)
        hidden = None
        captions = []
        
        for _ in range(max_len):
            output, hidden = self.decoder(features.unsqueeze(1), hidden)
            predicted_word_idx = output.argmax(2)
            word = vocab.idx2word[predicted_word_idx.item()]
            captions.append(word)
            if word == '<END>':
                break
        
        return ' '.join(captions)

# Cargar el modelo
def load_model(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Model loaded from {filename}")

# Cargar la imagen
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    return image

# Transformación de la imagen
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Evaluación de BLEU, ROUGE y METEOR en todo el conjunto de validación
def evaluate_model(model, val_loader, vocab, device):
    model.eval()
    
    # Variables para las métricas
    generated_captions = []
    ground_truth_captions = []

    # Bucle de evaluación
    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)

            # Generamos las descripciones
            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)  # Añadir un batch dimension
                # Generamos la descripción para la imagen
                generated_caption = model.generate_caption(image, vocab)
                generated_captions.append(generated_caption.split())  # Tokenizamos la descripción generada

                # Obtenemos la descripción correcta (ground truth)
                ground_truth_caption = captions[i]  # Esto asume que `captions` es un tensor de índices
                # Convertimos los índices de `captions` a palabras usando el vocabulario
                ground_truth_words = [vocab.idx2word[idx.item()] for idx in ground_truth_caption]
                ground_truth_captions.append([ground_truth_words])  # Necesitamos una lista de listas para BLEU

    # Evaluar las métricas
    bleu1 = bleu.compute(predictions=generated_captions, references=ground_truth_captions, max_order=1)
    bleu2 = bleu.compute(predictions=generated_captions, references=ground_truth_captions, max_order=2)
    
    rouge_results = rouge.compute(predictions=generated_captions, references=ground_truth_captions)
    
    meteor_results = meteor.compute(predictions=generated_captions, references=ground_truth_captions)
    
    # Mostrar las métricas
    print(f"BLEU-1: {bleu1['bleu']*100:.1f}%, BLEU-2: {bleu2['bleu']*100:.1f}%")
    print(f"ROUGE-L: {rouge_results['rougeL']*100:.1f}%")
    print(f"METEOR: {meteor_results['meteor']*100:.1f}%")
    
    # Devuelve el resultado como un string
    return f"BLEU-1: {bleu1['bleu']*100:.1f}%, BLEU-2: {bleu2['bleu']*100:.1f}%, ROUGE-L: {rouge_results['rougeL']*100:.1f}%, METEOR: {meteor_results['meteor']*100:.1f}%"

# Función para probar una imagen aleatoria
def test_random_image(model, vocab, image_dir="path_to_images"):
    import random
    # Cargar una imagen aleatoria
    image_path = random.choice([f"{image_dir}/{img}" for img in os.listdir(image_dir)])
    
    # Cargar y preprocesar la imagen
    transform = get_image_transform()
    image = load_image(image_path, transform).unsqueeze(0).to(device)  # Añadir un batch dimension
    
    # Generar la descripción
    model.eval()
    with torch.no_grad():
        caption = model.generate_caption(image, vocab)
    
    print(f"Generated Caption for the image {image_path}: {caption}")
    
    # Mostrar la imagen y la descripción generada
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Generated Caption: {caption}")
    plt.axis('off')
    plt.show()

# Entrenamiento y evaluación (configuración del modelo y vocabulario)
def main():
    # Cargar vocabulario y modelo
    vocab = load_vocab(r"Challenge 3\Image_Captioning\RESNET18_LSTM_vocab.pkl")
    model = EncoderDecoderModel(embed_size=256, hidden_size=512, vocab_size=len(d.dataset.vocab.word2idx)).to(device)
    load_model(r"Challenge 3\Image_Captioning\RESNET18_LSTM_model.pth", model)

    # Evaluar el modelo
    bleu, rouge, meteor = evaluate_model(model, d.test_loader, vocab)

    # Evaluar el modelo con una imagen aleatoria
    #test_random_image(model, vocab, "path_to_images")

if __name__ == "__main__":
    main()
