# Camouflaged Target Detection (CTD)

This project focuses on the detection of camouflaged military vehicles using a combination of YOLOv8 for object detection and CLIP for zero-shot classification. It includes scripts for training a baseline model, fine-tuning on high-resolution images, exporting the model to ONNX, and a Streamlit-based web application for interactive detection.

## Features

*   **Baseline Training**: Train a standard YOLOv8 model on your dataset.
*   **High-Res Fine-Tuning**: Fine-tune the model on higher-resolution images (1280px) to improve detection of small, camouflaged objects.
*   **ONNX Export**: Export the trained model to ONNX format with specific opset versioning for broad compatibility.
*   **Interactive Web App**: A Streamlit application that allows users to upload images, run detections, and classify objects using CLIP (Zero-Shot Learning).

## Installation

### Prerequisites

*   Python 3.8 or higher
*   CUDA-capable GPU (recommended for training)

### Dependencies

1.  Clone this repository or navigate to the project directory.
2.  Install the required Python packages:

```bash
pip install ultralytics streamlit transformers torch torchvision pillow opencv-python-headless numpy
```

*(Note: Ensure you satisfy the specific PyTorch requirements for your CUDA version if you plan to use GPU acceleration.)*

## Usage

### 1. Training the Baseline Model

To train the initial YOLOv8m baseline model:

```bash
python 01_train_baseline.py
```

This script uses `datasets/military_vehicles/data.yaml` and saves results to `runs/detect/baseline_run/`.

### 2. Fine-Tuning for High Resolution

 To fine-tune the baseline model on larger images (1280px) with memory optimizations (batch size 4):

```bash
python 02_finetune_hires.py
```

This picks up the best weights from the baseline run and saves the fine-tuned model to `runs/detect/high_res_finetune/`.

### 3. Exporting to ONNX

To export the fine-tuned model to ONNX format (opset 12):

```bash
python 03_export_model.py
```

### 4. Running the Web Application

To start the Streamlit interface for testing the model:

```bash
streamlit run app.py
```

This will launch a local web server (usually at `http://localhost:8501`) where you can upload images and see the detection results.

## Project Structure

*   `01_train_baseline.py`: Script for initial model training.
*   `02_finetune_hires.py`: Script for high-resolution fine-tuning.
*   `03_export_model.py`: Utility to export the model to ONNX.
*   `app.py`: The main Streamlit application file.
*   `datasets/`: Directory containing your dataset configuration and images.
*   `runs/`: Directory where training results (weights, logs) are saved.

## Credits

*   **YOLOv8** by Ultralytics
*   **CLIP** by OpenAI
