import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Dropout
import os
from sklearn.model_selection import train_test_split
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
        self.output_fc = nn.Linear(embed_dim, d.NUM_CHAR)
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
        self.caption_embedding = nn.Embedding(d.NUM_CHAR, embed_dim)  # Agregar esta línea para embellecer los captions
        
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

# Función principal para definir y entrenar el modelo
def train_image_captioning_model(train_loader,val_loader,device, lr):
    # Crear directorio para guardar gráficas y el modelo si no existen
    if not os.path.exists('results'):
        os.makedirs('results')
    
    EMBED_DIM = 512
    FF_DIM = 512
    EPOCHS = 50

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
    optimizer = optim.AdamW(caption_model.parameters(), lr= lr)

    # Configurar EarlyStopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=True)


    # Variables para almacenar métricas
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Variables para almacenar métricas
    train_bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    val_bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}

    # Entrenamiento del modelo
    for epoch in range(EPOCHS):
        caption_model.train()
        total_loss = 0
        total_acc = 0
        padding_idx = vocab['<PAD>']
        char2idx = dataset.vocab.char2idx
        idx2char = dataset.vocab.idx2char

        total_bleu_scores = {key: 0 for key in train_bleu_scores}

        for images, captions in train_loader:  # Usar el dataloader de entrenamiento
            images, captions = images.to(device), captions.to(device)
            
            optimizer.zero_grad()
            
            # Pasar las imágenes y los captions por el modelo
            predictions = caption_model(images, captions)
    
            
            # Calcular la pérdida
            #print(f"Predictions shape: {predictions.shape}")
            #print(f"Targets shape: {captions[:, 1:].shape}")

            loss = d.compute_loss(predictions[:, :-1], captions[:, 1:], padding_idx)  # Eliminar la última predicción

            loss.backward()
            optimizer.step()
            
            # Calcular la exactitud
            acc = d.compute_accuracy(predictions, captions[:, 1:])
            
            bleu_scores = d.compute_bleu(predictions, captions[:, 1:], idx2char, char2idx)
            for key in train_bleu_scores:
                total_bleu_scores[key] += bleu_scores[key]

            total_loss += loss.item()
            total_acc += acc
        
        # Promediar las métricas para la época
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)

        avg_bleu_scores = {key: total / len(train_loader) for key, total in total_bleu_scores.items()}
        for key in train_bleu_scores:
            train_bleu_scores[key].append(avg_bleu_scores[key])
        
        # Evaluación con el val_loader al final de cada época
        caption_model.eval()
        total_val_bleu_scores = {key: 0 for key in val_bleu_scores}

        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for images, captions in val_loader:  # Usar el dataloader de validación
                images, captions = images.to(device), captions.to(device)
                
                predictions = caption_model(images, captions)

                val_loss = d.compute_loss(predictions[:, :-1], captions[:, 1:], padding_idx)  # Eliminar la última predicción
                val_acc = d.compute_accuracy(predictions, captions[:, 1:])
                
                val_bleu_scores_epoch = d.compute_bleu(predictions, captions[:, 1:], idx2char, char2idx)
                for key in val_bleu_scores:
                    total_val_bleu_scores[key] += val_bleu_scores_epoch[key]
                
                total_val_loss += val_loss.item()
                total_val_acc += val_acc
        
        # Promediar las métricas para la época
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        
        # Guardar las métricas
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        avg_val_bleu_scores = {key: total / len(val_loader) for key, total in total_val_bleu_scores.items()}
        for key in val_bleu_scores:
            val_bleu_scores[key].append(avg_val_bleu_scores[key])
        
        # Imprimir el progreso por cada época
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}, "
              f"BLEU-1: {avg_bleu_scores['bleu_1']:.4f}, BLEU-2: {avg_bleu_scores['bleu_2']:.4f}, BLEU-3: {avg_bleu_scores['bleu_3']:.4f}, BLEU-4: {avg_bleu_scores['bleu_4']:.4f}, "
            
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}, "
              f"Validation BLEU-1: {avg_val_bleu_scores['bleu_1']:.4f}, BLEU-2: {avg_val_bleu_scores['bleu_2']:.4f}, BLEU-3: {avg_val_bleu_scores['bleu_3']:.4f}, BLEU-4: {avg_val_bleu_scores['bleu_4']:.4f}, ")

    # Guardar el modelo entrenado
    torch.save(caption_model.state_dict(), 'results/image_captioning_model_50.pth')
    print("Model saved!")

    # Graficar y guardar las métricas al final del entrenamiento
    d.plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, 
                   train_bleu_scores['bleu_1'], train_bleu_scores['bleu_2'], 
                   train_bleu_scores['bleu_3'], train_bleu_scores['bleu_4'])

# Llamar a la función de entrenamiento
if __name__ == '__main__':
    # Create the dataset
    dataset = d.FoodImageCaptionDataset(csv_path=d.csv_path, image_dir=d.image_dir, transform=d.image_transforms)
    vocab = dataset.vocab.char2idx

    # Dividir el índice de datos
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

    # Crear subconjuntos para entrenamiento, validación y prueba
    train_subset = d.SubsetFoodImageCaptionDataset(dataset, train_indices)
    val_subset = d.SubsetFoodImageCaptionDataset(dataset, val_indices)
    test_subset = d.SubsetFoodImageCaptionDataset(dataset, test_indices)

    # Crear los dataloaders
    train_loader = d.DataLoader(train_subset, batch_size=12, shuffle=True, collate_fn=d.collate_fn)
    val_loader = d.DataLoader(val_subset, batch_size=12, shuffle=False, collate_fn=d.collate_fn)
    test_loader = d.DataLoader(test_subset, batch_size=12, shuffle=False, collate_fn=d.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_image_captioning_model(train_loader,val_loader,device, 1e-4)