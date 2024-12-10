import pickle
import torch
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm



#LOAD HE VOCABULARY AND THE TEST SET

# Load vocabulary
def load_vocab(filename="vocab.pkl"):
    with open(filename, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {filename}")
    return vocab

# Load the vocabulary
vocab = load_vocab("vocab.pkl")

# Test DataLoader (Assuming `test_loader` is defined similarly to `val_loader`)
test_loader = d.test_loader  # Replace with the actual test DataLoader you are using



#MODIFYING THE EVALUATE_MODEL FUNCTION FOR THE TRANSFORMER FOR THE TEST SET

# Load evaluation metrics
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

# Evaluate the Transformer model
def evaluate_model(model, data_loader, vocab, device):
    model.eval()
    
    # Variables for the metrics
    generated_captions = []
    ground_truth_captions = []

    # Evaluation loop
    with torch.no_grad():
        for images, captions in tqdm(data_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Generate captions
            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)  # Add batch dimension
                generated_caption = model.generate(image, max_len=20, start_idx=vocab.word2idx["<START>"], end_idx=vocab.word2idx["<END>"])
                generated_captions.append(generated_caption)

                # Convert the ground truth caption to words
                ground_truth_caption = captions[i]  # Assuming captions are indices
                ground_truth_words = [vocab.idx2word[idx.item()] for idx in ground_truth_caption]
                ground_truth_captions.append([ground_truth_words])  # For BLEU, we need a list of lists

    # Compute BLEU, ROUGE, and METEOR scores
    bleu1 = bleu.compute(predictions=generated_captions, references=ground_truth_captions, max_order=1)
    bleu2 = bleu.compute(predictions=generated_captions, references=ground_truth_captions, max_order=2)
    
    rouge_results = rouge.compute(predictions=generated_captions, references=ground_truth_captions)
    meteor_score = meteor.compute(predictions=generated_captions, references=ground_truth_captions)

    # Print the evaluation results
    print(f"Average BLEU1: {bleu1['bleu']*100:.4f}%")
    print(f"Average BLEU2: {bleu2['bleu']*100:.4f}%")
    print(f"ROUGE-L: {rouge_results['rougeL']*100:.4f}%")
    print(f"METEOR: {meteor_score['meteor']*100:.4f}%")



# LOADING THE MODEL

# Load the model checkpoint
def load_model(model, checkpoint_path, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Model loaded from {checkpoint_path}")

# Initialize your model
model = TransformerImageCaptioning(embed_dim=256, num_heads=8, dense_dim=512, vocab_size=len(vocab.word2idx)).to(device)

# Load the model checkpoint
checkpoint_path = "path_to_model_checkpoint.pth"  # Replace with your checkpoint file path
load_model(model, checkpoint_path)



# EVALUATING THE MODEL ON THE TEST SET

# Evaluate the model on the test set
evaluate_model(model, test_loader, vocab, device)




