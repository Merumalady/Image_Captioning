import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# File paths
image_folder = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Images"
csv_file = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

# Load the dataset
df = pd.read_csv(csv_file)

# Add full image paths to the DataFrame
df['Image Path'] = df['Image Name'].apply(lambda x: os.path.join(image_folder, x))

# Filter rows where the image file doesn't exist
df = df[df['Image Path'].apply(os.path.exists)]

# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42) #20% FOR TESTING
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42) #SPLITS THE 20% INTO 50/50

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Define a PyTorch Dataset class
class FoodDataset(Dataset):
    def __init__(self, dataframe, transform=None, level='word'):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on an image.
            level (str): 'word' or 'char' level for the recipe title.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.level = level
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.dataframe.iloc[idx]['Image Path']
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Process recipe title
        recipe_title = self.dataframe.iloc[idx]['Recipe Name']
        if self.level == 'char':
            # Convert to character-level labels
            label = list(recipe_title)
        elif self.level == 'word':
            # Convert to word-level labels
            label = recipe_title.split()
        else:
            raise ValueError("Level must be 'word' or 'char'")
        
        return image, label

# Define transforms for the images
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create DataLoaders
batch_size = 32
train_dataset = FoodDataset(train_df, transform=image_transforms, level='char') #we chose char level
val_dataset = FoodDataset(val_df, transform=image_transforms, level='char')
test_dataset = FoodDataset(test_df, transform=image_transforms, level='char')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #ns si shuffle = true o = false
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Checking a batch of data
for images, labels in train_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels: {labels[:5]}")
    break
