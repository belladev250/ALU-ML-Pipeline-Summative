import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import warnings

warnings.filterwarnings('ignore')


class Predictor:
    """Make predictions with trained model"""
    
    def __init__(self, model_path, classes_path):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (.h5 file)
            classes_path: Path to saved classes (.pkl file)
        """
        print(f"TensorFlow version: {tf.__version__}")
        
        # Load the model with compile=False to avoid issues
        print("Loading model...")
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False  # Don't compile during load
            )
            print("✅ Model loaded successfully")
            
            # Recompile the model
            print("Recompiling model...")
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✅ Model compiled successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load classes
        print("Loading classes...")
        with open(classes_path, 'rb') as f:
            self.classes = pickle.load(f)
        
        # Auto-detect input size from model
        self.img_size = self._detect_input_size()
        
        print(f"✅ Model loaded: {model_path}")
        print(f"✅ Classes loaded: {list(self.classes)}")
        print(f"✅ Detected input size: {self.img_size}")
    
    def _detect_input_size(self):
        """Detect the correct input size from model architecture"""
        try:
            input_shape = self.model.input_shape
            print(f"Model input shape: {input_shape}")
            
            if input_shape and len(input_shape) >= 3:
                if len(input_shape) == 4:  # (batch, height, width, channels)
                    height, width = input_shape[1], input_shape[2]
                else:  # (height, width, channels)
                    height, width = input_shape[0], input_shape[1]
                
                if height is not None and width is not None and height > 0 and width > 0:
                    return (int(height), int(width))
        except Exception as e:
            print(f"Error detecting input size: {e}")
        
        # Default to common sizes if detection fails
        print("⚠️  Could not detect input size, using default (150, 150)")
        return (150, 150)
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.array: Preprocessed image with batch dimension
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, image_path):
        """
        Make prediction on single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Prediction results with class and confidence
        """
        img_array = self.preprocess_image(image_path)
        
        if img_array is None:
            return {'error': 'Failed to process image'}
        
        try:
            prediction = self.model.predict(img_array, verbose=0)
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
            return {
                'class': str(self.classes[class_idx]),
                'confidence': float(confidence),
                'probabilities': {
                    str(self.classes[i]): float(prediction[0][i]) 
                    for i in range(len(self.classes))
                }
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Prediction failed: {str(e)}'}
    
    def predict_batch(self, image_paths):
        """Make predictions on multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results
    
    def predict_from_array(self, img_array):
        """Make prediction from numpy array"""
        try:
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            
            prediction = self.model.predict(img_array, verbose=0)
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
            return {
                'class': str(self.classes[class_idx]),
                'confidence': float(confidence),
                'probabilities': {
                    str(self.classes[i]): float(prediction[0][i]) 
                    for i in range(len(self.classes))
                }
            }
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def get_classes(self):
        """Get list of class names"""
        return list(self.classes)
    
    def get_num_classes(self):
        """Get number of classes"""
        return len(self.classes)