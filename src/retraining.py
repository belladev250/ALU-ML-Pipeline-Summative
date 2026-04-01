import os
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import pickle

class Retrainer:
    """Simple retrainer for fruit classification"""
    
    def __init__(self, model_path, classes_path, img_size=(150, 150)):
        """
        Initialize retrainer
        
        Args:
            model_path: Path to current model
            classes_path: Path to classes file  
            img_size: Image size (height, width)
        """
        self.model_path = model_path
        self.classes_path = classes_path
        self.img_size = img_size
        
        try:
            # Load current model and classes
            self.model = tf.keras.models.load_model(model_path)
            with open(classes_path, 'rb') as f:
                self.classes = pickle.load(f)
            print(f"✅ Retrainer loaded with {len(self.classes)} classes")
        except Exception as e:
            print(f"❌ Error loading retrainer: {e}")
            self.model = None
            self.classes = []
    
    def extract_zip_data(self, zip_path, extract_dir='temp_retrain'):
        """Extract zip file containing new training data"""
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✅ Extracted data to {extract_dir}")
            return extract_dir
        except Exception as e:
            print(f"❌ Error extracting zip: {e}")
            return None
    
    def load_images_from_folder(self, folder_path):
        """Load images from directory structure"""
        images = []
        labels = []
        
        # Get all subdirectories (classes)
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            print(f"Loading {class_name}...", end=" ")
            count = 0
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, filename)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0
                        images.append(img_array)
                        labels.append(class_name)
                        count += 1
                    except Exception as e:
                        continue
            
            print(f"✅ {count} images")
        
        if len(images) == 0:
            print("❌ No images found!")
            return None, None
        
        return np.array(images), np.array(labels)
    
    def prepare_retrain_data(self, data_folder, test_size=0.2):
        """Prepare data for retraining"""
        X, y = self.load_images_from_folder(data_folder)
        
        if X is None:
            return None, None, None, None
        
        # Convert string labels to numeric using existing classes
        y_numeric = []
        valid_indices = []
        
        for i, label in enumerate(y):
            if label in self.classes:
                y_numeric.append(np.where(self.classes == label)[0][0])
                valid_indices.append(i)
        
        if len(y_numeric) == 0:
            print("❌ No matching classes found in new data!")
            return None, None, None, None
        
        # Filter images to only include valid classes
        X_filtered = X[valid_indices]
        y_numeric = np.array(y_numeric)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_filtered, y_numeric,
            test_size=test_size,
            random_state=42,
            stratify=y_numeric
        )
        
        print(f"✅ Retrain data prepared:")
        print(f"   Training: {len(X_train)} images")
        print(f"   Validation: {len(X_val)} images")
        
        return X_train, X_val, y_train, y_val
    
    def retrain_model(self, X_train, X_val, y_train, y_val, epochs=3, batch_size=32):
        """Retrain model with new data"""
        if self.model is None:
            print("❌ No model loaded for retraining")
            return None
        
        print("🚀 Starting retraining...")
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Simple training
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        print("✅ Retraining complete!")
        return history
    
    def save_retrained_model(self, backup=True):
        """Save retrained model"""
        if self.model is None:
            print("❌ No model to save")
            return False
        
        if backup and os.path.exists(self.model_path):
            backup_path = self.model_path.replace('.h5', '_backup.h5')
            shutil.copy(self.model_path, backup_path)
            print(f"✅ Old model backed up: {backup_path}")
        
        self.model.save(self.model_path)
        print(f"✅ Retrained model saved: {self.model_path}")
        return True
    
    def retrain_from_zip(self, zip_path, epochs=3, batch_size=32, cleanup=True):
        """Complete retraining pipeline from zip file"""
        try:
            print("🔄 Starting retraining pipeline...")
            
            # Extract
            extract_dir = self.extract_zip_data(zip_path)
            if extract_dir is None:
                return False
            
            # Prepare data
            X_train, X_val, y_train, y_val = self.prepare_retrain_data(extract_dir)
            if X_train is None:
                if cleanup and os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                return False
            
            # Retrain
            history = self.retrain_model(X_train, X_val, y_train, y_val, epochs, batch_size)
            
            if history is None:
                if cleanup and os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                return False
            
            # Save
            success = self.save_retrained_model(backup=True)
            
            # Cleanup
            if cleanup and os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
                print(f"✅ Temporary files cleaned up")
            
            return success
            
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
            # Cleanup on error
            if cleanup and os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            return False