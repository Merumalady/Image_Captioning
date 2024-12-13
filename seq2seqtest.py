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
from seq2seqbasic import ImageCaptioningModel, EncoderCNN

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

# Cargar el modelo
def load_model(model_path, cnn_type, rnn_type, embed_size, hidden_size, vocab_size, num_layers, device):
    # Inicializar el modelo con la arquitectura correcta
    model = ImageCaptioningModel(
        embed_size, hidden_size, vocab_size, cnn_type=cnn_type, rnn_type=rnn_type, num_layers=num_layers
    )
    # Cargar los pesos
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    vocab_size = len(d.dataset.vocab.word2idx)

    # Ruta del modelo a cargar
    cnn_type = "resnet18"
    rnn_type = "gru"
    model_name = f"{cnn_type.upper()}_{rnn_type.upper()}"
    model_path = f"Challenge 3\Image_Captioning\{model_name}_model.pth"

    # Cargar modelo
    model = load_model(
        model_path, cnn_type, rnn_type, embed_size, hidden_size, vocab_size, num_layers, device
    )

    evaluate_model(model, d.test_loader, d.dataset.vocab, device)

    # Evaluar el modelo con una imagen aleatoria
    #test_random_image(model, vocab, "path_to_images")


if __name__ == "__main__":
    main()

