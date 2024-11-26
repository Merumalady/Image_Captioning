from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd

class FoodImageCaptionDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): Ruta al archivo CSV.
            image_dir (str): Directorio donde se encuentran las imágenes.
            transform (callable, optional): Transformaciones aplicadas a las imágenes.
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image_Name']
        caption = self.data.iloc[idx]['Title']
        
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image {img_path} not found.")
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption


image_dir = r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Images\Food Images"
dataset = FoodImageCaptionDataset(csv_path=r"C:\Users\merit\OneDrive\Escritorio\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv", image_dir=image_dir)
#image_dir = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Images\Food Images"
#dataset = FoodImageCaptionDataset(csv_path=r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv", image_dir=image_dir)

sample_image, sample_caption = dataset[3]
sample_image.show()  
print(sample_caption)  
