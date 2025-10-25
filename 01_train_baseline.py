# File: 01_train_baseline.py

from ultralytics import YOLO
import os

def train_baseline_model():
    """
    Trains the baseline YOLOv8 model for multi-object detection.
    """
    print("--- Starting Baseline Model Training ---")

    # --- Configuration ---
    # Define the path to your dataset's YAML file.
    # IMPORTANT: Make sure this path is correct!
    dataset_yaml_path = os.path.join('datasets', 'military_vehicles', 'data.yaml')

    # Choose the YOLO model. 'yolov8m.pt' is a good balance of speed and accuracy.
    model_name = 'yolov8m.pt'
    
    # Training parameters
    epochs = 100
    image_size = 640
    patience = 20 # Early stopping: stops if no improvement after 20 epochs

    # --- Model Initialization ---
    # Load a pre-trained YOLOv8 model
    print(f"Loading pre-trained model: {model_name}")
    model = YOLO(model_name)

    # --- Model Training ---
    print(f"Starting training for {epochs} epochs with image size {image_size}...")
    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=image_size,
        patience=patience,
        project='runs/detect',  # Save results to 'runs/detect'
        name='baseline_run'       # Subfolder for this specific run
    )

    print("--- Baseline Training Finished ---")
    
    # The trained model and results are saved in 'runs/detect/baseline_run/'
    # The best performing model weights are saved as 'best.pt' inside that folder.
    print("Find your results in the 'runs/detect/baseline_run/' directory.")

if __name__ == '__main__':
    train_baseline_model()