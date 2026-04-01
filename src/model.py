

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class FruitClassificationModel:
    """Build and train CNN model with transfer learning"""
    
    def __init__(self, img_size=(150, 150), num_classes=2):
        """
        Initialize model builder
        
        Args:
            img_size: Input image size (height, width)
            num_classes: Number of output classes
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self, use_pretrained=True):
        """
        Build model with transfer learning
        
        Args:
            use_pretrained: Whether to use MobileNetV2 pretrained weights
            
        Returns:
            Compiled Keras model
        """
        if use_pretrained:
            # Load pre-trained MobileNetV2
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            # Build custom top layers
            model = models.Sequential([
                layers.Input(shape=(*self.img_size, 3)),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            # Build custom CNN from scratch
            model = models.Sequential([
                layers.Input(shape=(*self.img_size, 3)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Define callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        print("\n🚀 Starting training...\n")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n✅ Training complete!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input images (can be single image or batch)
            
        Returns:
            Predictions array
        """
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """
        Save model to file
        
        Args:
            filepath: Path to save model (.h5 file)
        """
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to model file (.h5)
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
    
    def get_model(self):
        """Get the Keras model object"""
        return self.model
    
    def get_history(self):
        """Get training history"""
        return self.history