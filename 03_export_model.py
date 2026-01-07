# File: 03_export_model.py (Revised with stable opset)

from ultralytics import YOLO
import os

def export_optimized_model():
    """
    Re-exports the model to ONNX with a specific opset version to
    ensure compatibility and fix the detection issue.
    """
    print("--- Starting Model Re-Export with Stable Opset ---")

    # --- Configuration ---
    fine_tuned_model_path = os.path.join('runs', 'detect', 'high_res_finetune2', 'weights', 'best.pt')

    # --- Model Loading and Export ---
    print(f"Loading model from: {fine_tuned_model_path}")
    model = YOLO(fine_tuned_model_path)

    # Export the model with a specific ONNX opset version for better compatibility
    print("Exporting to ONNX with opset=12...")
    model.export(
        format='onnx',
        imgsz=1280,
        half=True,
        opset=12  # ADDED: This is the key change for stability
    )

    print("--- Model Re-Export Finished ---")
    print(f"A new ONNX model has been created in: {os.path.dirname(fine_tuned_model_path)}")


if __name__ == '__main__':
    export_optimized_model()