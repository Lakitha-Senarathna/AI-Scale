import tensorflow as tf
import tf2onnx
from pathlib import Path
import traceback

def export2ONNX():
    try:
        # load model
        model_path = Path("ExportedModels/Tuned") / "tunedBestModel.keras"
        model = tf.keras.models.load_model(model_path)

        # define input specification of model
        input_spec = tf.TensorSpec((1, 224, 224, 3), tf.float32, name = 'input',)

        # define output path to save the model
        out_dir = Path("ExportedModels/Exported")

        output_path = out_dir / "onnxmodel.onnx"

        # convert to .onnx format
        tf2onnx.convert.from_keras(
            model,
            input_signature = input_spec,
            opset = 13,
            output_path = str(output_path)
        )

        print("Model saved")

    except Exception:
        print(f"Failed to save the exported .onnx model")
        traceback.print_exc()