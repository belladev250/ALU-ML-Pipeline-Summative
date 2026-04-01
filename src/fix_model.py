import tensorflow as tf
from tensorflow.keras.models import load_model

# Define a dummy InputLayer class to bypass 'batch_shape' argument
from tensorflow.keras.layers import InputLayer
class LegacyInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Remove batch_shape if present
        kwargs.pop('batch_shape', None)
        super().__init__(*args, **kwargs)

# Path to your old model
old_model_path = "./models/fruit_classifier_light.h5"
new_model_path = "./models/fruit_classifier_light_fixed.h5"

# Load the old model with custom_objects
model = load_model(
    old_model_path,
    custom_objects={'InputLayer': LegacyInputLayer}
)

# Save in modern compatible format
model.save(new_model_path)
print(f"✅ Model fixed and saved as: {new_model_path}")
