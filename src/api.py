"""
Flask REST API for fruit classification
Handles predictions, metrics, and retraining
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import threading
import pickle
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction import Predictor
from retraining import Retrainer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'zip', 'avif', 'webp', 'jfif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize predictor and retrainer with error handling
predictor = None
retrainer = None

def initialize_models():
    """Initialize predictor and retrainer"""
    global predictor, retrainer
    
    model_path = 'models/fruit_classifier_model.h5'
    classes_path = 'models/fruit_classes.pkl'
    
    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the model file exists before starting the API")
        return False
    
    if not os.path.exists(classes_path):
        print(f"❌ Classes file not found: {classes_path}")
        print("   Please ensure the classes file exists before starting the API")
        return False
    
    # Initialize predictor
    try:
        predictor = Predictor(model_path, classes_path)
        print("✅ Predictor loaded successfully")
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize retrainer
    try:
        retrainer = Retrainer(model_path, classes_path)
        print("✅ Retrainer loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Retrainer not loaded: {e}")
        # Retrainer is optional, so continue
    
    return True

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0
        self.start_time = time.time()
        self.retraining = False
    
    def add_request(self, response_time):
        self.request_count += 1
        self.total_response_time += response_time
    
    def get_metrics(self):
        uptime = time.time() - self.start_time
        avg_response_time = (self.total_response_time / self.request_count * 1000) if self.request_count > 0 else 0
        requests_per_minute = (self.request_count / (uptime / 60)) if uptime > 0 else 0
        
        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self.request_count,
            'average_response_time_ms': round(avg_response_time, 2),
            'requests_per_minute': round(requests_per_minute, 2),
            'retraining': self.retraining
        }

metrics = MetricsTracker()

def allowed_file(filename, types={'jpg', 'jpeg', 'png', 'zip', 'avif', 'webp', 'jfif'}):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in types

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': time.time()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Get API status"""
    return jsonify({
        'status': 'running',
        'model': 'fruit_classifier',
        'classes': predictor.get_classes() if predictor else [],
        'model_loaded': predictor is not None,
        'retrainer_loaded': retrainer is not None,
        'timestamp': time.time()
    }), 200

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    start_time = time.time()
    
    try:
        print("🔍 Starting prediction request...")
        
        if predictor is None:
            print("❌ Predictor is None - model not loaded")
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        if 'image' not in request.files:
            print("❌ No 'image' key in request.files")
            return jsonify({'error': 'No image provided. Use key "image" in form-data'}), 400
        
        file = request.files['image']
        print(f"📁 Received file: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename, {'jpg', 'jpeg', 'png', 'avif', 'webp', 'jfif'}):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Use jpg, jpeg, png, avif, webp, or jfif'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{time.time()}_{filename}")
        print(f"💾 Saving file to: {filepath}")
        
        file.save(filepath)
        print("✅ File saved successfully")
        
        # Make prediction
        print("🤖 Making prediction...")
        result = predictor.predict(filepath)
        print(f"✅ Prediction result: {result}")
        
        # Clean up
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print("✅ Temporary file cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️  Warning: Could not delete temp file: {cleanup_error}")
        
        # Track metrics
        response_time = time.time() - start_time
        metrics.add_request(response_time)
        
        print(f"✅ Prediction completed in {response_time:.2f}s")
        
        return jsonify({
            'prediction': result,
            'response_time_ms': round(response_time * 1000, 2),
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        print(f"❌ ERROR in /predict: {str(e)}")
        import traceback
        print("🔍 Full traceback:")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Make predictions on multiple images"""
    start_time = time.time()
    
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided. Use key "images" in form-data'}), 400
    
    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({'error': 'No images selected'}), 400
    
    try:
        results = []
        filepaths = []
        
        for file in files:
            if file and allowed_file(file.filename, {'jpg', 'jpeg', 'png', 'avif', 'webp', 'jfif'}):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{time.time()}_{filename}")
                file.save(filepath)
                filepaths.append(filepath)
                
                result = predictor.predict(filepath)
                results.append({
                    'filename': filename,
                    'prediction': result
                })
        
        # Clean up
        for filepath in filepaths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
        
        response_time = time.time() - start_time
        metrics.add_request(response_time)
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'response_time_ms': round(response_time * 1000, 2),
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get API metrics"""
    return jsonify(metrics.get_metrics()), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'classes': predictor.get_classes(),
        'num_classes': predictor.get_num_classes(),
        'model_file': 'models/fruit_classifier_model.h5',
        'input_size': list(predictor.img_size) + [3]
    }), 200

# ============================================================================
# RETRAINING ENDPOINTS
# ============================================================================

@app.route('/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining"""
    if retrainer is None:
        return jsonify({'error': 'Retrainer not initialized'}), 500
    
    if metrics.retraining:
        return jsonify({'error': 'Retraining already in progress'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Use key "file" in form-data'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, {'zip'}):
        return jsonify({'error': 'Invalid file type. Use zip'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{time.time()}_{filename}")
        file.save(filepath)
        
        # Get training parameters
        epochs = int(request.form.get('epochs', 20))
        batch_size = int(request.form.get('batch_size', 32))
        
        # Mark as retraining
        metrics.retraining = True
        
        # Run retraining in background thread
        def retrain_background():
            try:
                print(f"🔄 Starting retraining with epochs={epochs}, batch_size={batch_size}")
                success = retrainer.retrain_from_zip(
                    filepath,
                    epochs=epochs,
                    batch_size=batch_size,
                    cleanup=True
                )
                
                if success:
                    # Reload predictor with new model
                    global predictor
                    predictor = Predictor(
                        'models/fruit_classifier_model.h5',
                        'models/fruit_classes.pkl'
                    )
                    print("✅ Predictor reloaded with new model")
                else:
                    print("❌ Retraining failed")
                
                # Clean up zip file
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except:
                    pass
                    
            except Exception as e:
                print(f"❌ Retraining error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                metrics.retraining = False
        
        thread = threading.Thread(target=retrain_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Retraining started in background',
            'status': 'processing',
            'epochs': epochs,
            'batch_size': batch_size
        }), 202
        
    except Exception as e:
        metrics.retraining = False
        return jsonify({'error': str(e)}), 500

@app.route('/retrain-status', methods=['GET'])
def retrain_status():
    """Get retraining status"""
    return jsonify({
        'retraining': metrics.retraining,
        'timestamp': time.time()
    }), 200

# ============================================================================
# INFO ENDPOINTS
# ============================================================================

@app.route('/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'name': 'Fruit Classification API',
        'version': '1.0',
        'description': 'REST API for fruit image classification',
        'endpoints': {
            'POST /predict': 'Make prediction on single image (key: "image")',
            'POST /predict-batch': 'Make predictions on multiple images (key: "images")',
            'POST /retrain': 'Trigger model retraining (key: "file", type: zip)',
            'GET /metrics': 'Get API metrics',
            'GET /model-info': 'Get model information',
            'GET /health': 'Health check',
            'GET /status': 'API status',
            'GET /retrain-status': 'Retraining status',
            'GET /info': 'API information'
        }
    }), 200

@app.route('/', methods=['GET'])
def index():
    """API home page"""
    return jsonify({
        'message': 'Fruit Classification API',
        'status': 'running',
        'model_loaded': predictor is not None,
        'visit': '/info for endpoints'
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Fruit Classification API")
    print("=" * 60)
    
    # Initialize models
    models_ready = initialize_models()
    
    print("=" * 60)
    print(f"Model loaded: {predictor is not None}")
    print(f"Retrainer ready: {retrainer is not None}")
    print("=" * 60)
    
    if not models_ready:
        print("\n⚠️  WARNING: Model files not found!")
        print("   The API will start but predictions will fail.")
        print("   Please ensure these files exist:")
        print("   - models/fruit_classifier_model.h5")
        print("   - models/fruit_classes.pkl")
        print()
    
    print("\n📊 Starting server on http://0.0.0.0:5000")
    print("   Test with: curl http://localhost:5000/status\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False  # Prevent double initialization
    )