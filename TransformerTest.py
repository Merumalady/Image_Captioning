from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Dropout
import os
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from keras.callbacks import EarlyStopping

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


# Bloque Encoder del Transformer
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0.3):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            Dropout(dropout),  # Aplicar Dropout aquí
            nn.Linear(dense_dim, embed_dim),
            Dropout(dropout)   # Dropout adicional
        )
        self.dropout = Dropout(dropout)  # Dropout en atención

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_out))
        fc_out = self.fc(x)
        x = self.layer_norm2(x + fc_out)
        return x

# Bloque Decoder del Transformer
# En el bloque Decoder
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout=0.3):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            Dropout(dropout)
        )
        self.output_fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = Dropout(dropout)

    def forward(self, x, encoder_out):
        encoder_out = encoder_out.repeat(x.size(0), x.size(1), 1)
        cross_attn_out, _ = self.cross_attention(x, encoder_out, encoder_out)
        x = self.layer_norm2(x + self.dropout(cross_attn_out))
        fc_out = self.fc(x)
        x = self.layer_norm3(x + fc_out)
        logits = self.output_fc(x)
        return logits

# Modelo completo de Image Captioning
class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5, embed_dim=512):
        super(ImageCaptioningModel, self).__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim)  # Agregar esta línea para embellecer los captions
        
    # Modificar la extracción de características en la función forward del ImageCaptioningModel
    def forward(self, images, captions):
        # Extraer características de la imagen
        img_embed = self.cnn_model(images)  # [batch_size, embed_dim]

        # Ajustar la forma para que sea compatible con el encoder
        img_embed = img_embed.unsqueeze(1)  # [batch_size, 1, embed_dim]
        img_embed = img_embed.permute(1, 0, 2)  # [1, batch_size, embed_dim]

        # Pasar por el encoder
        encoder_out = self.encoder(img_embed)  # [1, batch_size, embed_dim]

        
        # Embedding de los captions antes de pasarlos al decoder
        caption_embeds = self.caption_embedding(captions)
        
        # Pasar las características por el decoder
        decoder_out = self.decoder(caption_embeds.permute(1, 0, 2), encoder_out)
        decoder_out = decoder_out.permute(1, 0, 2)

        
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

"""
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
        model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
        
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
    plt.savefig(f"without_bahdnau_test_accuracy_plot.png")
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
    plt.savefig(f"without_bahdnau_test_metrics_plot.png")
    plt.close()
"""


    # Variables necesarias
    #models_folder = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\models"
"""
    dataloader = test_loader  # Reemplaza con tu DataLoader de prueba
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx2word = dataset.vocab.idx2word  # Diccionario índice a palabra
    vocab = dataset.vocab.word2idx  # Diccionario palabra a índice
    padding_idx = vocab["<PAD>"]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    # Dividir el índice de datos
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

    test_subset = d.SubsetFoodImageCaptionDataset(dataset, test_indices)
    test_loader = d.DataLoader(test_subset, batch_size=12, shuffle=False, collate_fn=d.collate_fn)

    EMBED_DIM = 512
    FF_DIM = 512

    # Cargar el modelo CNN
    cnn_model = get_cnn_model()

    # Mover el modelo EfficientNet a la GPU si está disponible
    cnn_model.to(device)  # Asegúrate de mover el modelo a la misma GPU que las imágenes


    # Definir los bloques del Transformer
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=8)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=8)

    evaluate_models_in_folder(models_folder, dataloader, device, idx2word, vocab, padding_idx, loss_fn)
"""

if __name__ == "__main__":
    import data as d  # Assuming data.py contains your dataset and vocab loaders
    import random
    # Create the dataset
    dataset = d.FoodImageCaptionDataset(csv_path=d.csv_path, image_dir=d.image_dir, transform=d.image_transforms)
    vocab_size = len(dataset.vocab.word2idx)
    vocab = dataset.vocab.word2idx  # Diccionario palabra a índice

    EMBED_DIM = 512
    FF_DIM = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar el modelo CNN
    cnn_model = get_cnn_model()

    # Mover el modelo EfficientNet a la GPU si está disponible
    cnn_model.to(device)

    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=8)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=8)


    # Ruta de la carpeta donde están los modelos
    models_folder = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\models"
    image_folder = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\food_test"
    
    # Transform to be applied to each image (as defined in d.image_transforms)
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
                    show_image(Image.open(image_path), generated_caption, dataset.vocab.idx2word)


