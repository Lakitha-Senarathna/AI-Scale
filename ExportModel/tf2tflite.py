import tensorflow as tf
import traceback
from pathlib import Path

def convertModel2TFlite():

    try:
        # load tuned best .keras model
        model_path = Path("ExportedModels/Tuned") / "tunedBestModel.keras"

        model = tf.keras.models.load_model(model_path)

        # fixing input shape to (224, 224, 3) an batch size to 1 for tflite model
        input_specification = tf.keras.Input(shape = (224, 224, 3), batch_size = 1)
        input_fixed_model = tf.keras.Model(input_specification, model(input_specification))
        
        # convert to .tflite format
        converter = tf.lite.TFLiteConverter.from_keras_model(input_fixed_model)
        model_tflite = converter.convert()

        # define output path to save the model
        out_dir = Path("ExportedModels/Exported")
        
        output_path = out_dir / "tflitemodel.tflite"

        # save the model
        output_path.write_bytes(model_tflite)

        print("Print Model saved")

    except Exception as e:
        print(f"Failed to save the exported .tflite model")
        traceback.print_exc()