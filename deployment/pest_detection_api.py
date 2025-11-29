#!/usr/bin/env python3
"""
ONNX-based Pest Detection API
Uses ONNX Runtime instead of TFLite (works better on Windows)
"""

from __future__ import annotations

import io
import os
import time
import numpy as np
from typing import Dict, Any
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available. Install with: pip install onnxruntime")

app = Flask(__name__)

# Model configuration
ONNX_MODEL_PATH = None
CLASS_NAMES = []
session = None

def find_onnx_model() -> str:
    """Find ONNX model file"""
    base_dir = Path(__file__).resolve().parent
    
    # Priority order - check models/ directory first (for Heroku deployment)
    candidates = [
        base_dir / "models" / "best 2.onnx",
        base_dir / "models" / "best.onnx",
        base_dir / "models" / "best5.onnx",
        base_dir / "datasets" / "best 2.onnx",
        base_dir / "datasets" / "best.onnx",
        base_dir / "datasets" / "best5.onnx",
        base_dir / "pest_detection_ml" / "models" / "best.onnx",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    raise FileNotFoundError(
        f"ONNX model not found. Checked: {[str(c) for c in candidates]}"
    )

def load_onnx_model(model_path: str):
    """Load ONNX model"""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")
    
    print(f"📦 Loading ONNX model: {Path(model_path).name}")
    
    # Create inference session
    sess = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']  # Use CPU
    )
    
    # Get input/output details
    input_details = sess.get_inputs()[0]
    output_details = sess.get_outputs()[0]
    
    print(f"✅ Model loaded")
    print(f"   Input: {input_details.name}, Shape: {input_details.shape}")
    print(f"   Output: {output_details.name}, Shape: {output_details.shape}")
    
    return sess, input_details, output_details

# Initialize model
if ONNX_AVAILABLE:
    try:
        ONNX_MODEL_PATH = find_onnx_model()
        session, input_details, output_details = load_onnx_model(ONNX_MODEL_PATH)
        
        # Default class names (update based on your model)
        CLASS_NAMES = [
            "Rice_Bug",
            "White stem borer",
            "black-bug",
            "brown_hopper",
            "green_hopper",
        ]
    except Exception as e:
        print(f"⚠️  Could not load ONNX model: {e}")
        session = None
        input_details = None
        output_details = None
else:
    session = None
    input_details = None
    output_details = None

@app.get("/")
def index() -> Any:
    """Root endpoint - API information"""
    return jsonify({
        "name": "AgriShield Pest Detection API",
        "version": "1.0.0",
        "framework": "ONNX Runtime",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)"
        },
        "model_loaded": session is not None,
        "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none"
    })

@app.get("/health")
def health() -> Any:
    """Health check endpoint"""
    if not ONNX_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "ONNX Runtime not available",
            "install": "pip install onnxruntime"
        }), 500
    
    if session is None:
        return jsonify({
            "status": "error",
            "message": "ONNX model not loaded"
        }), 500
    
    return jsonify({
        "status": "ok",
        "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "framework": "ONNX Runtime"
    })

def preprocess_image(image: Image.Image, input_shape: tuple) -> np.ndarray:
    """Preprocess image for ONNX model"""
    # Get target size from input shape (usually [1, 3, H, W] or [1, H, W, 3])
    if len(input_shape) == 4:
        if input_shape[1] == 3:  # NCHW format
            h, w = input_shape[2], input_shape[3]
        else:  # NHWC format
            h, w = input_shape[1], input_shape[2]
    else:
        h, w = 512, 512  # Default
    
    # Resize
    img = image.resize((w, h))
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Handle format (ONNX usually expects NCHW: [1, 3, H, W])
    if len(img_array.shape) == 3:  # HWC
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def postprocess_output(output_data: np.ndarray, conf_threshold: float = 0.15) -> Dict[str, int]:
    """Postprocess ONNX model output to get pest counts"""
    counts = {name: 0 for name in CLASS_NAMES}
    
    # ONNX YOLO output format: [batch, num_detections, 6]
    # Where 6 = [x, y, w, h, confidence, class_id]
    if len(output_data.shape) == 3:
        detections = output_data[0]  # Remove batch dimension
    elif len(output_data.shape) == 2:
        detections = output_data
    else:
        return counts
    
    for detection in detections:
        if len(detection) >= 6:
            conf = float(detection[4])
            class_id = int(detection[5])
            
            if conf >= conf_threshold and 0 <= class_id < len(CLASS_NAMES):
                counts[CLASS_NAMES[class_id]] += 1
    
    return counts

@app.post("/detect")
def detect() -> Any:
    """Pest detection endpoint using ONNX Runtime"""
    if not ONNX_AVAILABLE or session is None:
        return jsonify({
            "error": "ONNX model not available",
            "message": "Install ONNX Runtime or load model"
        }), 500
    
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400
    
    t0 = time.time()
    
    # Preprocess
    input_shape = input_details.shape if input_details.shape else [1, 3, 512, 512]
    input_data = preprocess_image(img, input_shape)
    
    # Run inference
    input_name = input_details.name
    output = session.run([output_details.name], {input_name: input_data})
    output_data = output[0]
    
    dt = time.time() - t0
    
    # Postprocess
    counts = postprocess_output(output_data, conf_threshold=0.15)
    
    # Pesticide recommendations
    pesticide_recs = {
        "Rice_Bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
        "green_hopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance.",
        "brown_hopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
        "black-bug": "Carbaryl dust or fipronil bait at tillering; field sanitation recommended.",
    }
    
    recommendations = {k: v for k, v in pesticide_recs.items() if counts.get(k, 0) > 0}
    
    return jsonify({
        "status": "success",
        "pest_counts": counts,
        "recommendations": recommendations,
        "inference_time_ms": round(dt * 1000, 1),
        "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none",
        "framework": "ONNX Runtime"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    print(f"Starting ONNX-based Pest Detection API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

