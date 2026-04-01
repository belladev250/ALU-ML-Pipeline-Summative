"""
Load testing with Locust for Fruit Classification API
Tests the API under heavy load and measures response times
"""

from locust import HttpUser, task, between
import random
import io
from PIL import Image


class FruitClassificationUser(HttpUser):
    """Simulate user interactions with the API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Create dummy images for testing"""
        self.image_bytes_list = []
        
        # Generate 5 different dummy images
        for i in range(5):
            img = Image.new('RGB', (150, 150), color=(random.randint(0, 255), 
                                                        random.randint(0, 255), 
                                                        random.randint(0, 255)))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            self.image_bytes_list.append(img_bytes)

    @task(3)
    def health_check(self):
        """Health check endpoint"""
        self.client.get(
            "/health",
            name="/health"
        )

    @task(10)
    def predict_single(self):
        """Make single prediction"""
        img_bytes = random.choice(self.image_bytes_list)
        img_bytes.seek(0)
        
        self.client.post(
            "/predict",
            files={'image': ('test.jpg', img_bytes, 'image/jpeg')},
            name="/predict"
        )

    @task(2)
    def get_metrics(self):
        """Get API metrics"""
        self.client.get(
            "/metrics",
            name="/metrics"
        )

    @task(1)
    def model_info(self):
        """Get model information"""
        self.client.get(
            "/model-info",
            name="/model-info"
        )

    @task(2)
    def status(self):
        """Get API status"""
        self.client.get(
            "/status",
            name="/status"
        )

    @task(1)
    def info(self):
        """Get API info"""
        self.client.get(
            "/info",
            name="/info"
        )


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         Fruit Classification API - Load Testing               ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Usage:
    
    1. Start the API server first:
       python src/api.py
    
    2. In another terminal, run Locust:
       locust -f locustfile.py --host=http://localhost:5000 -u 50 -r 10 -t 5m
    
    Parameters:
       -u 50       = 50 concurrent users
       -r 10       = Spawn 10 users per second
       -t 5m       = Run test for 5 minutes
    
    Examples:
    
    Light Load (10 users):
       locust -f locustfile.py --host=http://localhost:5000 -u 10 -r 5 -t 5m
    
    Medium Load (50 users):
       locust -f locustfile.py --host=http://localhost:5000 -u 50 -r 10 -t 5m
    
    Heavy Load (100 users):
       locust -f locustfile.py --host=http://localhost:5000 -u 100 -r 20 -t 5m
    
    Ultra Heavy Load (200 users):
       locust -f locustfile.py --host=http://localhost:5000 -u 200 -r 40 -t 10m
    
    With different container counts:
    
    1 Container:  locust -f locustfile.py --host=http://localhost:5000 -u 50 -r 10 -t 5m
    2 Containers: docker-compose up -d --scale api=2 && locust ...
    3 Containers: docker-compose up -d --scale api=3 && locust ...
    """)