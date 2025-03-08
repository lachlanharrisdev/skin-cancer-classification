let model = null;
let video = document.getElementById('camera');
let resultDiv = document.getElementById('result');

// Load TensorFlow.js model
async function loadModel() {
    model = await tf.loadGraphModel('tfjs_model/model.json');
    console.log("Model loaded");
}

// Access webcam
async function setupCamera() {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" }, 
        audio: false 
    });
    await new Promise(resolve => video.onloadedmetadata = resolve);
}

// Preprocess image for model
function preprocessImage(image) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(image)
            .resizeBilinear([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims();
        return tensor;
    });
}

// Make prediction
async function predict() {
    const image = tf.tidy(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 224, 224);
        return preprocessImage(canvas);
    });

    const prediction = await model.predict(image);
    const probabilities = await prediction.data();
    tf.dispose([image, prediction]);

    const benignProb = probabilities[0];
    const malignantProb = probabilities[1];
    const confidence = Math.max(benignProb, malignantProb) * 100;
    const diagnosis = benignProb > malignantProb ? 'Benign' : 'Malignant';

    // Display results
    resultDiv.className = diagnosis.toLowerCase();
    resultDiv.innerHTML = `
        <h3>Result: ${diagnosis}</h3>
        <p>Confidence: ${confidence.toFixed(1)}%</p>
        <p>Benign Probability: ${(benignProb * 100).toFixed(1)}%</p>
        <p>Malignant Probability: ${(malignantProb * 100).toFixed(1)}%</p>
        <h4>Recommendation:</h4>
        ${getRecommendation(malignantProb)}
    `;
}

function getRecommendation(malignantProb) {
    if (malignantProb > 0.7) {
        return "High risk of malignancy. Consult a dermatologist immediately.";
    } else if (malignantProb > 0.4) {
        return "Moderate risk. Schedule a professional examination within 1-2 weeks.";
    } else {
        return "Low risk. Monitor for changes in size, color, or shape. Re-check monthly.";
    }
}

// Initialize app
async function init() {
    await loadModel();
    await setupCamera();
    document.getElementById('captureBtn').addEventListener('click', predict);
}

init();
