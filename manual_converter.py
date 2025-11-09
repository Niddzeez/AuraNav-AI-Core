import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import os
import shutil

# --- Configuration ---
DETECTOR_ONNX_PATH = r"runs\detect\train2\weights\best.onnx"
SEGMENTER_ONNX_PATH = r"runs\segment\train9\weights\best.onnx"

DETECTOR_TFLITE_PATH = r"runs\detect\train2\weights\best_float32.tflite"
SEGMENTER_TFLITE_PATH = r"runs\segment\train9\weights\best_float32.tflite"
# ---------------------

def convert_model(onnx_path, tflite_path):
    try:
        if not os.path.exists(onnx_path):
            print(f"ERROR: ONNX file not found at {onnx_path}. Skipping.")
            return

        print(f"--- Starting conversion for: {onnx_path} ---")
        
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        print(f"Loaded ONNX model: {onnx_path}")
        
        # Check the model for any issues
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")
        
        # Prepare the ONNX model for TensorFlow backend
        print("Preparing TensorFlow representation...")
        tf_rep = prepare(onnx_model)
        print("TensorFlow representation ready.")
        
        # Export the model as a TensorFlow SavedModel
        print("Exporting to temporary TensorFlow SavedModel...")
        tf_rep.export_graph("temp_savedmodel")
        print("Temporary SavedModel created.")
        
        # Convert the SavedModel to a TFLite file
        print("Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model("temp_savedmodel")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print("TFLite conversion complete.")
        
        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"\n[SUCCESS] Saved TFLite model to: {tflite_path}\n")

    except Exception as e:
        print(f"An error occurred during conversion of {onnx_path}: {e}")
    
    finally:
        # Clean up the temporary SavedModel directory
        if os.path.exists("temp_savedmodel"):
            shutil.rmtree("temp_savedmodel")
            print("Cleaned up temporary files.")

if __name__ == "__main__":
    convert_model(DETECTOR_ONNX_PATH, DETECTOR_TFLITE_PATH)
    convert_model(SEGMENTER_ONNX_PATH, SEGMENTER_TFLITE_PATH)
    print("--- All conversions complete ---")