import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Dropout
import os
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from keras.callbacks import EarlyStopping
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


# Bloque Encoder del Transformer
class TransformerEncoderBlock_antiguo(nn.Module):
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



# Función principal para definir y entrenar el modelo
def train_image_captioning_model_bahdnau(train_loader,val_loader,device, vocab, EPOCHS = 30 ,lr= 1e-4, dropout=0.3):
    # Crear directorio para guardar gráficas y el modelo si no existen
    if not os.path.exists('results'):
        os.makedirs('results')
    

    # Cargar el modelo CNN
    cnn_model = get_cnn_model()

    # Mover el modelo EfficientNet a la GPU si está disponible
    cnn_model.to(device)  # Asegúrate de mover el modelo a la misma GPU que las imágenes

    encoder = TransformerEncoderBlock(embed_dim=512, dense_dim=512, num_heads=8, dropout= dropout)
    name_model = f'word_level_{EPOCHS}_{lr}_{dropout}_bahdnau_batch'

    decoder = BahdanauDecoder(embed_dim=512, encoder_dim=512, decoder_dim=512, attention_dim=256, vocab_size=len(vocab), dropout= dropout)

    caption_model = ImageCaptioningModel(cnn_model, encoder, decoder).to(device)

    # Configurar el optimizador
    optimizer = optim.AdamW(caption_model.parameters(), lr= lr, weight_decay=1e-5)

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
        padding_idx = vocab["<PAD>"]
        char2idx = dataset.vocab.word2idx
        idx2char = dataset.vocab.idx2word

        total_bleu_scores = {key: 0 for key in train_bleu_scores}

        for batch_idx, (images, captions) in enumerate(train_loader):  # Usar el dataloader de entrenamiento
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
            acc = d.compute_accuracy(predictions, captions[:, 1:], padding_idx)
            
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
            for batch_idx, (images, captions) in enumerate(val_loader):  # Usar el dataloader de validación
                images, captions = images.to(device), captions.to(device)
                
                predictions = caption_model(images, captions)

                val_loss = d.compute_loss(predictions[:, :-1], captions[:, 1:], padding_idx)  # Eliminar la última predicción
                val_acc = d.compute_accuracy(predictions, captions[:, 1:], padding_idx)
                
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
              f"Validation BLEU-1: {avg_val_bleu_scores['bleu_1']:.4f}, BLEU-2: {avg_val_bleu_scores['bleu_2']:.4f}, BLEU-3: {avg_val_bleu_scores['bleu_3']:.4f}, BLEU-4: {avg_val_bleu_scores['bleu_4']:.4f}")

    # Guardar el modelo entrenado
    torch.save(caption_model.state_dict(), f"results/{name_model}.pth")
    print("Model saved!")

    # Graficar y guardar las métricas al final del entrenamiento
    d.plot_metrics(name_model, train_losses, train_accuracies, val_losses, val_accuracies, 
                   train_bleu_scores['bleu_1'], train_bleu_scores['bleu_2'], 
                   train_bleu_scores['bleu_3'], train_bleu_scores['bleu_4'])

if __name__ == "__main__":
    import data as d  # Assuming data.py contains your dataset and vocab loaders
    # Create the dataset
    dataset = d.FoodImageCaptionDataset(csv_path=d.csv_path, image_dir=d.image_dir, transform=d.image_transforms)

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

    vocab_size = len(dataset.vocab.word2idx)
    vocab = dataset.vocab.word2idx

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_image_captioning_model_bahdnau(train_loader,val_loader,device, vocab, EPOCHS = 100,lr= 1e-3, dropout=0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_image_captioning_model_bahdnau(train_loader,val_loader,device, vocab, EPOCHS = 100,lr= 1e-4, dropout=0.5)




