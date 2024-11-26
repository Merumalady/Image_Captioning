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

        self.data['Image_Path'] = self.data['Image_Name'].apply(lambda x: os.path.join(self.image_dir, f"{x}.jpg"))
        self.data = self.data[self.data['Image_Path'].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Obtener el nombre de la imagen y la descripción (caption)
        img_path = self.data.iloc[idx]['Image_Path']
        caption = self.data.iloc[idx]['Title']
        
        # Cargar la imagen
        image = Image.open(img_path).convert("RGB")
        
        # Aplicar transformaciones, si existen
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
