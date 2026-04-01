
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class FruitDataProcessor:
    """Process and load fruit images for model training"""
    
    def __init__(self, img_size=(150, 150)):
        """
        Initialize processor
        
        Args:
            img_size: Tuple of (height, width) for resizing images
        """
        self.img_size = img_size
        self.classes = None
        
    def load_images_from_folder(self, folder_path):
        """
        Load images from directory
        
        Args:
            folder_path: Path to folder containing class subdirectories
            
        Returns:
            tuple: (images array, labels array)
        """
        images = []
        labels = []
        
        for fruit_class in sorted(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, fruit_class)
            if not os.path.isdir(class_path):
                continue
            
            print(f"Loading {fruit_class}...", end=" ")
            count = 0
            
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, filename)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0
                        images.append(img_array)
                        labels.append(fruit_class)
                        count += 1
                    except Exception as e:
                        pass
            
            print(f"✅ {count} images")
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, data_dir, test_size=0.2, val_size=0.2, random_state=42):
        """
        Load and split data into train/val/test sets
        
        Args:
            data_dir: Path to dataset directory
            test_size: Proportion for test set (default 0.2)
            val_size: Proportion for validation set (default 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X, y = self.load_images_from_folder(data_dir)
        
        self.classes = np.unique(y)
        print(f"\n✅ Classes: {list(self.classes)}")
        print(f"✅ Total images: {len(X)}\n")
        
        # Convert string labels to numeric indices
        y_numeric = np.array([np.where(self.classes == label)[0][0] for label in y])
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_numeric, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_numeric
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            random_state=random_state, 
            stratify=y_temp
        )
        
        print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.array: Preprocessed image
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            return img_array
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def get_classes(self):
        """Get class names"""
        return self.classes
    
    def get_num_classes(self):
        """Get number of classes"""
        return len(self.classes) if self.classes is not None else 0