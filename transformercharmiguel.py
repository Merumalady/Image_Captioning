import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from nltk.translate.bleu_score import sentence_bleu
from efficientnet_pytorch import EfficientNet
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import datachar as d  # Asegúrate de que este módulo esté disponible

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
    def __init__(self, embed_dim, dense_dim, num_heads):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        
    def forward(self, x):
        # Cambiar la forma de los tensores para que sean [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim] -> [seq_len, batch_size, embed_dim]

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)

        # Capa Feed-forward
        fc_out = self.fc(x)
        x = self.layer_norm2(x + fc_out)

        return x


# Bloque Decoder del Transformer
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.output_fc = nn.Linear(embed_dim, d.NUM_CHAR)

    def forward(self, x, encoder_out):
        print(f"x shape: {x.shape}")
        print(f"encoder_out shape: {encoder_out.shape}")

        # Cambiar la forma de x para que sea [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim] -> [seq_len, batch_size, embed_dim]

        # Cambiar la forma de encoder_out para que sea [seq_len, batch_size, embed_dim]
        encoder_out = encoder_out.permute(1, 0, 2)  # Asegúrate de que encoder_out esté en la forma correcta

        # Cross-attention con la salida del encoder
        cross_attn_out, _ = self.cross_attention(x, encoder_out, encoder_out)
        x = self.layer_norm2(x + cross_attn_out)

        # Capa Feed-forward
        fc_out = self.fc(x)
        x = self.layer_norm3(x + fc_out)

        # Capa de salida
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
        self.caption_embedding = nn.Embedding(d.NUM_CHAR, embed_dim)  # Agregar esta línea para embellecer los captions
        
    # Modificar la extracción de características en la función forward del ImageCaptioningModel
    def forward(self, images, captions):
        # Extraer características de la imagen
        img_embed = self.cnn_model(images)  # Forma: [batch_size, embed_dim]
        
        # Añadir una dimensión de secuencia, para que tenga forma [seq_len=1, batch_size, embed_dim]
        img_embed = img_embed.unsqueeze(0)  # Forma: [1, batch_size, embed_dim]
        
        # Pasar las características por el encoder
        encoder_out = self.encoder(img_embed)
        
        # Embedding de los captions antes de pasarlos al decoder
        caption_embeds = self.caption_embedding(captions)
        
        # Pasar las características por el decoder
        decoder_out = self.decoder(caption_embeds, encoder_out)
        
        return decoder_out


# Función de cálculo de pérdida
def compute_loss(predictions, targets, padding_idx=d.char2idx['<PAD>']):
    # Solo se calcula la pérdida para los tokens que no son padding
    mask = (targets != padding_idx).float()
    loss = F.cross_entropy(predictions.reshape(-1, d.NUM_CHAR), targets.reshape(-1), reduction='none')
    loss = loss * mask.reshape(-1)
    return loss.sum() / mask.sum()



# Función para calcular la exactitud (accuracy)
def compute_accuracy(predictions, targets, padding_idx=d.char2idx['<PAD>']):
    # Solo se calcula la exactitud para los tokens que no son padding
    pred_ids = predictions.argmax(dim=-1)
    mask = (targets != padding_idx).float()
    correct = (pred_ids == targets) * mask
    accuracy = correct.sum() / mask.sum()
    return accuracy.item()


# Función para calcular BLEU-1 y BLEU-2
def compute_bleu(predictions, targets):
    pred_ids = predictions.argmax(dim=-1)
    pred_tokens = [d.idx2char[i.item()] for i in pred_ids[0]]
    target_tokens = [d.idx2char[i.item()] for i in targets[0]]
    
    # BLEU-1 y BLEU-2
    bleu_1 = sentence_bleu([target_tokens], pred_tokens, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu([target_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0))
    
    return bleu_1, bleu_2


# Función principal para definir y entrenar el modelo
def train_image_captioning_model(device):
    # Crear directorio para guardar gráficas y el modelo si no existen
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Definir los parámetros
    IMAGE_SIZE = (224, 224)
    TEXT_MAX_LEN = 201
    EMBED_DIM = 512
    FF_DIM = 512
    EPOCHS = 30

    # Cargar el modelo CNN
    cnn_model = get_cnn_model()

    # Mover el modelo EfficientNet a la GPU si está disponible
    cnn_model.to(device)  # Asegúrate de mover el modelo a la misma GPU que las imágenes


    # Definir los bloques del Transformer
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=2)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=4)

    # Crear el modelo de captioning
    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    caption_model.to(device)
    # Configurar el optimizador
    optimizer = optim.Adam(caption_model.parameters(), lr=1e-4)

    # Configurar EarlyStopping para evitar sobreajuste
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

    # Variables para almacenar métricas
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    bleu_1_scores = []
    bleu_2_scores = []

    # Entrenamiento del modelo
    for epoch in range(EPOCHS):
        caption_model.train()
        total_loss = 0
        total_acc = 0
        total_bleu_1 = 0
        total_bleu_2 = 0
        for images, captions in d.train_loader:  # Usar el dataloader de entrenamiento
            images, captions = images.to(device), captions.to(device)
            
            optimizer.zero_grad()
            
            # Pasar las imágenes y los captions por el modelo
            predictions = caption_model(images, captions)
            
            # Calcular la pérdida
            print(f"Predictions shape: {predictions.shape}")
            print(f"Targets shape: {captions[:, 1:].shape}")

            loss = compute_loss(predictions, captions[:, 1:])  # Usar captions desplazados
            loss.backward()
            optimizer.step()
            
            # Calcular la exactitud
            acc = compute_accuracy(predictions, captions[:, 1:])
            
            # Calcular BLEU-1 y BLEU-2
            bleu_1, bleu_2 = compute_bleu(predictions, captions[:, 1:])
            
            total_loss += loss.item()
            total_acc += acc
            total_bleu_1 += bleu_1
            total_bleu_2 += bleu_2
        
        # Promediar las métricas para la época
        avg_train_loss = total_loss / len(d.train_loader)
        avg_train_acc = total_acc / len(d.train_loader)
        avg_bleu_1 = total_bleu_1 / len(d.train_loader)
        avg_bleu_2 = total_bleu_2 / len(d.train_loader)
        
        # Evaluación con el val_loader al final de cada época
        caption_model.eval()
        total_val_loss = 0
        total_val_acc = 0
        total_val_bleu_1 = 0
        total_val_bleu_2 = 0
        with torch.no_grad():
            for images, captions in d.val_loader:  # Usar el dataloader de validación
                images, captions = images.to(device), captions.to(device)
                
                predictions = caption_model(images, captions)
                
                val_loss = compute_loss(predictions, captions[:, 1:])
                val_acc = compute_accuracy(predictions, captions[:, 1:])
                
                bleu_1, bleu_2 = compute_bleu(predictions, captions[:, 1:])
                
                total_val_loss += val_loss.item()
                total_val_acc += val_acc
                total_val_bleu_1 += bleu_1
                total_val_bleu_2 += bleu_2
        
        # Promediar las métricas para la época
        avg_val_loss = total_val_loss / len(d.val_loader)
        avg_val_acc = total_val_acc / len(d.val_loader)
        
        # Guardar las métricas
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        bleu_1_scores.append(avg_bleu_1)
        bleu_2_scores.append(avg_bleu_2)
        
        # Imprimir el progreso por cada época
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}, "
              f"BLEU-1: {avg_bleu_1:.4f}, BLEU-2: {avg_bleu_2:.4f}")

    # Guardar el modelo entrenado
    torch.save(caption_model.state_dict(), 'results/image_captioning_model.pth')
    print("Model saved!")

    # Graficar y guardar las métricas al final del entrenamiento
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, bleu_1_scores, bleu_2_scores)


# Función para generar y guardar gráficas
def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, bleu_1_scores, bleu_2_scores):
    plt.figure(figsize=(16, 8))
    
    # Gráfico de Pérdida
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de Exactitud
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Gráfico de BLEU-1
    plt.subplot(2, 2, 3)
    plt.plot(bleu_1_scores, label='BLEU-1')
    plt.title('BLEU-1')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-1')
    plt.legend()

    # Gráfico de BLEU-2
    plt.subplot(2, 2, 4)
    plt.plot(bleu_2_scores, label='BLEU-2')
    plt.title('BLEU-2')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-2')
    plt.legend()
    
    # Guardar las gráficas
    plt.tight_layout()
    plt.savefig('results/metrics.png')
    plt.close()


# Llamar a la función de entrenamiento
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_image_captioning_model(device)
