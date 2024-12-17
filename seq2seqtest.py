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
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, cnn_type="resnet18"):
        super(EncoderCNN, self).__init__()
        
        if cnn_type.startswith("resnet"):
            resnet_models = {
                "resnet18": models.resnet18
            }
            backbone = resnet_models[cnn_type](pretrained=True)
            modules = list(backbone.children())[:-1]  # Eliminar la capa de clasificación
            self.cnn = nn.Sequential(*modules)
            in_features = backbone.fc.in_features
        
        elif cnn_type.startswith("vgg"):
            vgg_models = {
                "vgg16": models.vgg16
            }
            backbone = vgg_models[cnn_type](pretrained=True)
            modules = list(backbone.features)
            self.cnn = nn.Sequential(*modules, nn.AdaptiveAvgPool2d((7, 7)))
            in_features = 512 * 7 * 7
        
        self.fc = nn.Linear(in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        with torch.no_grad():
            features = self.cnn(images).view(images.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, rnn_type="gru"):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.feature_fc = nn.Linear(embed_size, hidden_size)  # 256 -> 512
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]  # Eliminar el último token para predecir el siguiente
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        inputs = self.feature_fc(inputs)  # Transformar al tamaño de hidden_size
        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs)
        return outputs

    def sample(self, features, max_len=20, start_idx=2, end_idx=3):
        # Añadir dimensión de secuencia a las características
        inputs = features.unsqueeze(0)  # Cambia de (embed_size) a (1, embed_size)
        inputs = self.feature_fc(inputs)  # Cambia de (1, embed_size) a (1, hidden_size)
        inputs = inputs.unsqueeze(1)  # Añade dimensión temporal: (1, 1, hidden_size)

        hidden = None
        outputs = []
        
        for _ in range(max_len):
            # Paso de la RNN
            out, hidden = self.rnn(inputs, hidden)
            out = self.fc(out.squeeze(1))  # Quitar dimensión temporal para pasar a softmax
            predicted = out.argmax(1)  # Obtener la palabra predicha
            outputs.append(predicted.item())
            
            if predicted.item() == end_idx:  # Si se alcanza el token <END>, detener la generación
                break
            
            # Preparar la siguiente entrada: incrustar el token predicho y añadir dimensión temporal
            inputs = self.embed(predicted).unsqueeze(1)  # Cambia de (batch_size, embed_size) a (batch_size, 1, embed_size)
            inputs = self.feature_fc(inputs)  # Ajustar nuevamente para hidden_size
        
        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, cnn_type="resnet18", rnn_type="gru", num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size, cnn_type=cnn_type)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=num_layers, rnn_type=rnn_type)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, image, vocab, max_len=20):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            features = self.encoder(image)  # Extract features
            inputs = features.unsqueeze(0)  # Add batch dimension

            # Initialize hidden state (GRU needs a hidden state)
            hidden = torch.zeros(1, 1, self.decoder.rnn.hidden_size).to(image.device)

            # Start the caption with the <START> token
            predicted_caption = []
            for _ in range(max_len):
                # Pass the current input and hidden state to the GRU
                out, hidden = self.decoder.rnn(inputs, hidden)

                # Get the predicted word (take argmax over the output)
                output = self.decoder.fc(out.squeeze(1))
                predicted_idx = output.argmax(1).item()

                predicted_caption.append(predicted_idx)

                # If we hit the end token, stop generating
                if predicted_idx == vocab.word2idx["<END>"]:
                    break

                # Prepare the next input: embed the predicted token
                inputs = self.decoder.embed(predicted_idx).unsqueeze(1)
                inputs = self.decoder.feature_fc(inputs)  # Ensure the shape matches GRU input

        # Decode the word indices to get the caption as a string
        caption = vocab.decode(predicted_caption)
        return " ".join([word for word in caption if word not in ["<START>", "<END>", "<PAD>"]])



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
    dataset = d.FoodImageCaptionDataset(csv_path=d.csv_path, image_dir=d.image_dir, transform=d.image_transforms)

    embed_size = 256
    hidden_size = 512
    num_layers = 2
    vocab_size = len(dataset.vocab.word2idx)

    # Ruta del modelo a cargar
    cnn_type = "resnet18"
    rnn_type = "gru"
    model_name = f"{cnn_type.upper()}_{rnn_type.upper()}"
    model_path = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\RESNET18_GRU_model.pth"

    # Cargar modelo
    model = load_model(
        model_path, cnn_type, rnn_type, embed_size, hidden_size, vocab_size, num_layers, device
    )
    #Variables necesarias
    train_indices, val_indices = d.train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_indices, test_indices = d.train_test_split(val_indices, test_size=0.5, random_state=42)

    test_subset = d.SubsetFoodImageCaptionDataset(dataset, test_indices)
    test_loader = d.DataLoader(test_subset, batch_size=12, shuffle=False, collate_fn=d.collate_fn)

    evaluate_model(model, test_loader, dataset.vocab, device)

    # Evaluar el modelo con una imagen aleatoria
    #test_random_image(model, vocab, "path_to_images")


if __name__ == "__main__":
    main()

