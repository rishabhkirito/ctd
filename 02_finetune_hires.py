# File: 02_finetune_hires.py (Updated with memory fix)

from ultralytics import YOLO
import os

def finetune_high_resolution():
    """
    Fine-tunes the baseline model on higher-resolution images to improve
    precision and generalization for camouflage patterns.
    """
    print("--- Starting High-Resolution Fine-Tuning (with memory optimization) ---")

    # --- Configuration ---
    dataset_yaml_path = os.path.join('datasets', 'military_vehicles', 'data.yaml')
    baseline_model_path = os.path.join('runs', 'detect', 'baseline_run', 'weights', 'best.pt')

    # --- MODIFIED PARAMETERS ---
    epochs = 50
    image_size = 1280  # REDUCED: A compromise for 6GB VRAM. Still a big improvement over 640.
    batch_size = 4    # ADDED: Manually set a small batch size to conserve VRAM.
    patience = 10

    # --- Model Initialization ---
    print(f"Loading model for fine-tuning from: {baseline_model_path}")
    model = YOLO(baseline_model_path)

    # --- Model Fine-Tuning ---
    print(f"Starting fine-tuning for {epochs} epochs with image size {image_size} and batch size {batch_size}...")
    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size, # Pass the batch size argument here
        patience=patience,
        project='runs/detect',
        name='high_res_finetune'
    )

    print("--- High-Resolution Fine-Tuning Finished ---")
    print("Find your new, improved model in the 'runs/detect/high_res_finetune/' directory.")

if __name__ == '__main__':
    finetune_high_resolution()