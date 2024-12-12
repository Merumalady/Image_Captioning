import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import pickle

bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

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
        if rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
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


    

def evaluate_model(model, data_loader, vocab):
    model.eval()
    total_bleu1 = 0
    total_bleu2 = 0
    total_rouge = 0
    total_meteor = 0
    count = 0  # Para contar el número de iteraciones

    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            features = model.encoder(images)
            
            # Generar salidas para cada imagen
            outputs = [model.decoder.sample(feature, start_idx=vocab.word2idx["<START>"], end_idx=vocab.word2idx["<END>"]) for feature in features]
            predicted = [vocab.decode(output) for output in outputs]

            predicted = [
                " ".join([word for word in caption if word not in ["<START>", "<END>", "<PAD>"]])  # Unir las palabras después de eliminarlas
                for caption in predicted
            ]

            #print(predicted, '\n')
            
            # Eliminar el token <START> y <END> de las referencias
            references = [vocab.decode(cap.tolist()) for cap in captions]

            references = [
                " ".join([word for word in ref if word not in ["<START>", "<END>", "<PAD>"]])  # Unir las palabras después de eliminarlas
                for ref in references
            ]

            #print(references, '\n')
            # Calcular BLEU (1 y 2)
            bleu1 = bleu.compute(predictions=predicted, references=references, max_order=1)
            bleu2 = bleu.compute(predictions=predicted, references=references, max_order=2)
            total_bleu1 += bleu1["bleu"]
            total_bleu2 += bleu2["bleu"]

            # Calcular ROUGE-L
            res_r = rouge.compute(predictions=predicted, references=references)
            total_rouge += res_r['rougeL']
            
            # Calcular METEOR
            res_m = meteor.compute(predictions=predicted, references=references)
            total_meteor += res_m['meteor']  # Promediamos METEOR

            count += 1  # Incrementamos el contador de iteraciones

    # Calcular promedios de cada métrica
    avg_bleu1 = total_bleu1 / count
    avg_bleu2 = total_bleu2 / count
    avg_rouge = total_rouge / count
    avg_meteor = total_meteor / count

    print(f"Average BLEU1 Score: {avg_bleu1*100:.4f}%")
    print(f"Average BLEU2 Score: {avg_bleu2*100:.4f}%")
    print(f"Average ROUGE-L Score: {avg_rouge*100:.4f}%")
    print(f"Average METEOR Score: {avg_meteor*100:.4f}%")


# Función de entrenamiento
def train_model(model, name, train_loader, val_loader, vocab, device ,num_epochs=20, learning_rate=1e-3, optimizer_type="adam"):
    criterion = CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD
    }
    optimizer = optimizers[optimizer_type](model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, captions in tqdm(train_loader):
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Evaluación después de cada época
        evaluate_model(model, val_loader, vocab)

# Función para guardar el vocabulario
def save_vocab(vocab, name):
    filename = f"Challenge 3\Image_Captioning\{name}_vocab.pkl"
    with open(filename, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved at {filename}")

if __name__ == "__main__":
    import data as d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    vocab_size = len(d.dataset.vocab.word2idx)

    # Definición de combinaciones de modelos
    combinations = [
        ("resnet18", "gru"), ("resnet18", "lstm"), ("vgg16", "gru")
    ]

    for cnn_type, rnn_type in combinations:
        name = f"{cnn_type.upper()}_{rnn_type.upper()}"
        print(f"Training {cnn_type.upper()} + {rnn_type.upper()}...")
        model = ImageCaptioningModel(
            embed_size, hidden_size, vocab_size, cnn_type=cnn_type, rnn_type=rnn_type, num_layers=num_layers
        ).to(device)

        train_model(
            model, name, d.train_loader, d.val_loader, d.dataset.vocab, device, 
            num_epochs=20, learning_rate=1e-3, optimizer_type="adamw"
        )

        torch.save(model.state_dict(), f"Challenge 3\Image_Captioning\{name}_model.pth")
        save_vocab(d.dataset.vocab, name)