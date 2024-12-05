import data as d

image_dir = r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Images\Food Images"
dataset = d.FoodImageCaptionDataset(csv_path=r"C:\Users\migue\OneDrive\Escritorio\UAB INTELIGENCIA ARTIFICIAL\Tercer Any\3A\Vision and Learning\Challenge 3\Image_Captioning\archive\Food Ingredients and Recipe Dataset with Image Name Mapping.csv", image_dir=image_dir)


# Spliting the dataset
train_data, validation_data, test_data = d.train_val_split(dataset)
print(f"Total number of samples: {len(dataset)}")
print(f"----> Number of training samples: {len(train_data)}")
print(f"----> Number of validation samples: {len(validation_data)}")
print(f"----> Number of test samples: {len(test_data)}")