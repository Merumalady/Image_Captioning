from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Dropout
import os
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

# Definir el modelo CNN basado en EfficientNet
def get_cnn_model():
    # Cargar EfficientNetB0 preentrenado
    base_model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Eliminar la cabeza de clasificación
    base_model._fc = nn.Identity()
    
    # Crear una capa lineal para reducir la dimensión de salida
    cnn_model = nn.Sequential(
        base_model,
        nn.Linear(1280, 512)  # Reducir la dimensión de 1280 a 512
    )
    
    return cnn_model


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.3):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Aplicar Dropout aquí
            nn.BatchNorm1d(dense_dim),  # Aquí usamos BatchNorm1d para normalizar las características
            nn.Linear(dense_dim, embed_dim),
            nn.Dropout(dropout)   # Dropout adicional
        )
        self.dropout = nn.Dropout(dropout)  # Dropout en atención

    def forward(self, x):
        # Cambiar la forma del tensor de [seq_len, batch_size, embed_dim] a [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        
        attn_out, _ = self.attention(x, x, x)
        
        # Añadir el resultado de la atención al tensor de entrada y aplicar la capa de normalización
        x = self.layer_norm1(x + self.dropout(attn_out))
        
        # Aplanar el tensor para que BatchNorm1d reciba un tensor de forma [batch_size * seq_len, embed_dim]
        x_flattened = x.view(-1, x.size(-1))  # Aplanar [batch_size * seq_len, embed_dim]
        
        fc_out = self.fc(x_flattened)  # Pasar el tensor a través de las capas completamente conectadas
        
        # Volver a darle la forma original de [batch_size, seq_len, embed_dim]
        x = fc_out.view(x.size(0), x.size(1), -1)
        
        # Aplicar la segunda capa de normalización
        x = self.layer_norm2(x + fc_out)
        
        return x

#BAHDANAU ATTENTION
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state).unsqueeze(1)
        combined_states = torch.tanh(u_hs + w_ah)
        attention_scores = self.A(combined_states).squeeze(2)
        alpha = F.softmax(attention_scores, dim=1)
        attention_weights = features * alpha.unsqueeze(2)
        context = attention_weights.sum(dim=1)
        return alpha, context

        

class BahdanauDecoder(nn.Module):
    def __init__(self, embed_dim, encoder_dim, decoder_dim, attention_dim, vocab_size, dropout=0.3):
        super(BahdanauDecoder, self).__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        #self.fc = nn.Linear(decoder_dim, vocab_size)
        self.fc = nn.Sequential(
            nn.Linear(decoder_dim, vocab_size),
            nn.BatchNorm1d(vocab_size)  # Añadir BatchNorm
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions, encoder_out):
        batch_size = captions.size(0)
        hidden_state, cell_state = self.init_hidden_state(batch_size, encoder_out.size(-1))

        outputs = []
        for t in range(captions.size(1)):
            embeddings = self.embedding(captions[:, t])
            alpha, context = self.attention(encoder_out, hidden_state)
            lstm_input = torch.cat((embeddings, context), dim=1)
            hidden_state, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))
            output = self.fc(self.dropout(hidden_state))
            outputs.append(output.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def init_hidden_state(self, batch_size, encoder_dim):
        hidden_state = torch.zeros(batch_size, encoder_dim).to(next(self.parameters()).device)
        cell_state = torch.zeros(batch_size, encoder_dim).to(next(self.parameters()).device)
        return hidden_state, cell_state

    
class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        # Extraer características de la imagen
        img_features = self.cnn_model(images)  # [batch_size, 512]
        
        # Reestructurar para ser compatible con el encoder
        img_features = img_features.unsqueeze(1)  # [batch_size, 1, 512]
        encoder_out = self.encoder(img_features)  # [batch_size, 1, 512]
        
        # Decodificar con atención
        decoder_out = self.decoder(captions, encoder_out)
        return decoder_out

# Función para preprocesar y cargar una imagen aleatoria
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)



# Función para generar el caption
import torch
import torch.nn.functional as F

def generate_caption(image_tensor, model, device, max_len=10, idx2word=None, top_k=10):
    
    # Poner el modelo en modo evaluación
    model.eval()

    # Mover la imagen al dispositivo correcto
    image_tensor = image_tensor.to(device)

    # Generar la caption (inicializar la secuencia con el token <START>)
    caption = [vocab["<START>"]]
    
    # Realizar la predicción paso a paso (decodificación autoregresiva)
    with torch.no_grad():
        for _ in range(max_len):
            # Convertir la secuencia de índices en un tensor de entrada
            input_tensor = torch.tensor(caption).unsqueeze(0).to(device)
            
            # Pasar la imagen y la secuencia de entrada por el modelo
            predictions = model(image_tensor, input_tensor)
            
            # Aplicar softmax para obtener probabilidades
            probs = F.softmax(predictions[:, -1, :], dim=-1)
            
            # Obtener los índices de las top-k palabras más probables
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Hacer muestreo entre las top-k palabras (en lugar de solo escoger la más probable)
            top_k_probs = top_k_probs.squeeze(0).cpu().numpy()  # Convertir a numpy para muestreo
            top_k_indices = top_k_indices.squeeze(0).cpu().numpy()
            
            # Muestrear una palabra entre las top-k
            predicted_idx = top_k_indices[torch.multinomial(torch.tensor(top_k_probs), 1)].item()
            
            # Si se genera el token <END>, terminamos la generación
            if predicted_idx == vocab["<END>"]:
                break
            
            # Añadir el índice de la palabra predicha a la secuencia
            caption.append(predicted_idx)

    # Convertir los índices de las palabras a las palabras reales
    generated_caption = [idx2word.get(idx, "<UNK>") for idx in caption[1:]]  # Ignorar el token <START>

    return generated_caption

import matplotlib.pyplot as plt

def show_image(image_tensor, caption, idx2word):
    plt.imshow(image_tensor)
    plt.title(' '.join(caption))
    plt.axis('off')
    plt.show()


import torch
import os
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import numpy as np

# Función para evaluar un modelo
def evaluate_model(model, dataloader, device, idx2word, vocab, loss_fn, padding_idx):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    bleu_scores = {f'BLEU-{i}': [] for i in range(1, 5)}
    rouge_scores = []

    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)

            # Predicción
            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]
            
            # Cálculo de pérdida
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

            # Cálculo de exactitud
            pred_ids = outputs.argmax(dim=-1)
            mask = (targets != padding_idx).float()
            correct = (pred_ids == targets) * mask
            accuracy = correct.sum() / mask.sum()
            total_accuracy += accuracy.item()

            # Cálculo de métricas
            for i in range(len(images)):
                pred_caption = [idx2word[idx.item()] for idx in pred_ids[i]]
                target_caption = [idx2word[idx.item()] for idx in targets[i] if idx.item() != padding_idx]
                for n in range(1, 5):
                    bleu_scores[f'BLEU-{n}'].append(sentence_bleu([target_caption], pred_caption, weights=[1/n]*n))
                rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                rouge_scores.append(rouge.score(" ".join(target_caption), " ".join(pred_caption))["rougeL"].fmeasure)


    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_bleu = {k: np.mean(v) for k, v in bleu_scores.items()}
    avg_rouge = np.mean(rouge_scores)
    return avg_loss, avg_accuracy, avg_bleu, avg_rouge

# Función principal
def evaluate_models_in_folder(models_folder, dataloader, device, idx2word, vocab, padding_idx, loss_fn):
    results = {}

    for model_file in os.listdir(models_folder):
        model = ImageCaptioningModel(cnn_model, encoder, decoder).to(device)
        
        if model_file.endswith(".pth"):

            model_path = os.path.join(models_folder, model_file)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            loss, accuracy, bleu_scores, rouge = evaluate_model(
                model, dataloader, device, idx2word, vocab, loss_fn, padding_idx
            )
            results[model_file] = {
                "loss": loss,
                "accuracy": accuracy,
                **bleu_scores,
                "rouge": rouge,
            }

    # Graficar resultados
    plot_results(results, model_file)

def plot_results(results, model_name):
    models = list(results.keys())

    # Gráfico de exactitud
    accuracies = [results[m]["accuracy"] for m in models]
    plt.figure(figsize=(10, 5))
    plt.plot(models, accuracies, label="Test Accuracy", marker="o")
    plt.xticks(rotation=0, ha='right')
    plt.title("Test Accuracy per Model")
    plt.legend()
    plt.savefig(f"bahdnau_test_accuracy_plot_0.5.png")
    plt.close()

    # Gráfico de métricas BLEU, ROUGE y METEOR
    bleu_scores = [[results[m][f"BLEU-{i}"] for m in models] for i in range(1, 5)]
    rouge_scores = [results[m]["rouge"] for m in models]

    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.plot(models, bleu_scores[i], label=f"BLEU-{i+1}", marker="o")
    plt.plot(models, rouge_scores, label="ROUGE", marker="o")
    plt.xticks(rotation=0, ha='right')
    plt.title("BLEU, ROUGE per Model")
    plt.legend()
    plt.savefig(f"bahdnau_test_metrics_plot_0.5.png")
    plt.close()


if __name__ == "__main__":

    import data as d  # Assuming data.py contains your dataset and vocab loaders
    import random
    # Create the dataset
    dataset = d.FoodImageCaptionDataset(csv_path=d.csv_path, image_dir=d.image_dir, transform=d.image_transforms)
    vocab_size = len(dataset.vocab.word2idx)
    vocab = dataset.vocab.word2idx  # Diccionario palabra a índice

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar el modelo CNN
    cnn_model = get_cnn_model()

    # Mover el modelo EfficientNet a la GPU si está disponible
    cnn_model.to(device)

    encoder = TransformerEncoderBlock(embed_dim=512, dense_dim=512, num_heads=8, dropout= 0.5)
    decoder = BahdanauDecoder(embed_dim=512, encoder_dim=512, decoder_dim=512, attention_dim=256, vocab_size=len(vocab), dropout= 0.5)

    # Ruta de la carpeta donde están los modelos
    models_folder = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\models_bahdnau_0.5"
    #image_folder = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\food_test"
    
    """# Transform to be applied to each image (as defined in d.image_transforms)
    transform = d.image_transforms  # Ensure you have this transform defined correctly in your 'data.py'

    # Iterate through all image files in the directory
    for image_file in os.listdir(image_folder):
        # Check if the file is an image (you can add more formats if needed)
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(image_folder, image_file)
            
            # Load and preprocess the image
            image_tensor = load_and_preprocess_image(image_path, transform).to(device)

            # Iterate over the models
            for model_file in os.listdir(models_folder):
                if model_file.endswith(".pth"):  # Check if it's a model file
                    model_path = os.path.join(models_folder, model_file)
                    model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)

                    # Generate the caption for the image
                    generated_caption = generate_caption(image_tensor, model, device, max_len=10, idx2word=dataset.vocab.idx2word)
                    print(f"Generated Caption for {model_file}: ", " ".join(generated_caption))

                    # Display the image and caption
                    show_image(Image.open(image_path), generated_caption, dataset.vocab.idx2word)"""


    #Variables necesarias
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

    test_subset = d.SubsetFoodImageCaptionDataset(dataset, test_indices)
    test_loader = d.DataLoader(test_subset, batch_size=12, shuffle=False, collate_fn=d.collate_fn)

  
    dataloader = test_loader  # Reemplaza con tu DataLoader de prueba
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx2word = dataset.vocab.idx2word  # Diccionario índice a palabra
    vocab = dataset.vocab.word2idx  # Diccionario palabra a índice
    padding_idx = vocab["<PAD>"]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    evaluate_models_in_folder(models_folder, dataloader, device, idx2word, vocab, padding_idx, loss_fn)

