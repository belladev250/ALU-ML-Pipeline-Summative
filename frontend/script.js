// Configuration
const API_URL = 'http://localhost:5000';

// Update metrics every 2 seconds
setInterval(updateMetrics, 2000);
updateMetrics();

// Load model info on page load
loadModelInfo();
setInterval(loadModelInfo, 10000);

// Update footer timestamp
updateFooterTimestamp();
setInterval(updateFooterTimestamp, 1000);

// ============================================================================
// METRICS FUNCTIONS
// ============================================================================

async function updateMetrics() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        const data = await response.json();

        // Update uptime
        const uptime = Math.floor(data.uptime_seconds);
        const hours = Math.floor(uptime / 3600);
        const minutes = Math.floor((uptime % 3600) / 60);
        const seconds = uptime % 60;
        document.getElementById('uptime').textContent = 
            `${hours.toString().padStart(2, '0')} : ${minutes.toString().padStart(2, '0')} : ${seconds.toString().padStart(2, '0')}`;

        // Update other metrics
        document.getElementById('total-requests').textContent = data.total_requests;
        document.getElementById('avg-response').textContent = data.average_response_time_ms.toFixed(2);
        document.getElementById('req-per-min').textContent = data.requests_per_minute.toFixed(2);

        // Update retraining status
        const retrainingEl = document.getElementById('retrain-status');
        if (data.retraining) {
            retrainingEl.textContent = '⚙️ PROCESSING';
            retrainingEl.className = 'metric-value status-idle';
        } else {
            retrainingEl.textContent = 'IDLE';
            retrainingEl.className = 'metric-value status-good';
        }

    } catch (error) {
        console.error('Error fetching metrics:', error);
        document.getElementById('model-status').textContent = '✗ OFFLINE';
        document.getElementById('model-status').className = 'metric-value status-bad';
    }
}

// ============================================================================
// MODEL INFO FUNCTIONS
// ============================================================================

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_URL}/model-info`);
        const data = await response.json();

        const infoDiv = document.getElementById('model-info');
        infoDiv.innerHTML = `
            <div style="display: grid; gap: 15px;">
                <div>
                    <h4>📋 Model Details</h4>
                    <p><strong>Classes:</strong> ${data.classes.join(', ')}</p>
                    <p><strong>Number of Classes:</strong> ${data.num_classes}</p>
                    <p><strong>Input Size:</strong> ${data.input_size.join(' × ')}</p>
                    <p><strong>Model File:</strong> ${data.model_file}</p>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('model-info').innerHTML = 
            '<p class="error">Unable to load model information</p>';
    }
}

// ============================================================================
// PREDICTION FUNCTIONS
// ============================================================================

function previewImage() {
    const fileInput = document.getElementById('image-input');
    const preview = document.getElementById('image-preview');

    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            preview.classList.add('active');
        };
        reader.readAsDataURL(fileInput.files[0]);
    }
}

async function makePrediction() {
    const fileInput = document.getElementById('image-input');
    const resultDiv = document.getElementById('prediction-result');

    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<div class="result-box error">❌ Please select an image</div>';
        return;
    }

    resultDiv.innerHTML = '<div class="result-box processing"><div class="loading"></div> Processing...</div>';

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();

        const confidence = (data.prediction.confidence * 100).toFixed(2);
        const probsHtml = Object.entries(data.prediction.probabilities)
            .map(([fruit, prob]) => `<p>${fruit}: ${(prob * 100).toFixed(2)}%</p>`)
            .join('');

        resultDiv.innerHTML = `
            <div class="result-box success">
                <h3>✅ Prediction Result</h3>
                <p><strong>Fruit Type:</strong> ${data.prediction.class}</p>
                <p><strong>Confidence:</strong> ${confidence}%</p>
                <p><strong>Response Time:</strong> ${data.response_time_ms.toFixed(2)}ms</p>
                <hr style="margin: 10px 0;">
                <h4>📊 Probabilities:</h4>
                ${probsHtml}
            </div>
        `;
    } catch (error) {
        resultDiv.innerHTML = `<div class="result-box error">❌ Error: ${error.message}</div>`;
    }

    fileInput.value = '';
    document.getElementById('image-preview').classList.remove('active');
}

// ============================================================================
// BATCH PREDICTION FUNCTIONS
// ============================================================================

async function batchPredict() {
    const fileInput = document.getElementById('batch-images');
    const resultDiv = document.getElementById('batch-result');

    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<div class="result-box error">❌ Please select images</div>';
        return;
    }

    resultDiv.innerHTML = '<div class="result-box processing"><div class="loading"></div> Processing batch...</div>';

    const formData = new FormData();
    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('images', fileInput.files[i]);
    }

    try {
        const response = await fetch(`${API_URL}/predict-batch`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Batch prediction failed');
        }

        const data = await response.json();

        let html = `
            <div class="result-box success">
                <h3>✅ Batch Prediction Complete</h3>
                <p><strong>Images Processed:</strong> ${data.count}</p>
                <p><strong>Response Time:</strong> ${data.response_time_ms.toFixed(2)}ms</p>
            </div>
            <div class="batch-results">
        `;

        data.predictions.forEach(pred => {
            const conf = (pred.prediction.confidence * 100).toFixed(2);
            html += `
                <div class="batch-item">
                    <h4>${pred.prediction.class}</h4>
                    <p><strong>Confidence:</strong> ${conf}%</p>
                    <small>${pred.filename}</small>
                </div>
            `;
        });

        html += '</div>';
        resultDiv.innerHTML = html;

    } catch (error) {
        resultDiv.innerHTML = `<div class="result-box error">❌ Error: ${error.message}</div>`;
    }

    fileInput.value = '';
}

// ============================================================================
// RETRAINING FUNCTIONS
// ============================================================================

async function retrainModel() {
    const fileInput = document.getElementById('retrain-file');
    const statusDiv = document.getElementById('retrain-status-msg');

    if (!fileInput.files.length) {
        statusDiv.innerHTML = '<div class="result-box error">❌ Please select a ZIP file</div>';
        return;
    }

    statusDiv.innerHTML = '<div class="result-box processing"><div class="loading"></div> Uploading and starting retraining...</div>';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch(`${API_URL}/retrain`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok || response.status === 202) {
            statusDiv.innerHTML = `
                <div class="result-box success">
                    <h3>✅ Retraining Started</h3>
                    <p><strong>Status:</strong> ${data.status}</p>
                    <p>${data.message}</p>
                    <p>The model will be automatically reloaded when complete.</p>
                </div>
            `;

            // Poll retraining status
            pollRetrainingStatus();
        } else {
            statusDiv.innerHTML = `<div class="result-box error">❌ Error: ${data.error}</div>`;
        }

    } catch (error) {
        statusDiv.innerHTML = `<div class="result-box error">❌ Error: ${error.message}</div>`;
    }

    fileInput.value = '';
}

async function pollRetrainingStatus() {
    let isRetraining = true;
    let pollCount = 0;
    const maxPolls = 600; // 10 minutes with 1 second intervals

    while (isRetraining && pollCount < maxPolls) {
        try {
            const response = await fetch(`${API_URL}/retrain-status`);
            const data = await response.json();

            isRetraining = data.retraining;

            if (!isRetraining) {
                document.getElementById('retrain-status-msg').innerHTML = `
                    <div class="result-box success">
                        <h3>✅ Retraining Complete</h3>
                        <p>Model has been successfully retrained and reloaded.</p>
                    </div>
                `;
            }

        } catch (error) {
            console.error('Error checking retraining status:', error);
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
        pollCount++;
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function updateFooterTimestamp() {
    const now = new Date();
    const timestamp = now.toLocaleString();
    document.getElementById('footer-timestamp').textContent = timestamp;
}

// Check API connection on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ Connected to API');
        }
    } catch (error) {
        console.error('❌ Cannot connect to API. Make sure server is running at', API_URL);
    }
});