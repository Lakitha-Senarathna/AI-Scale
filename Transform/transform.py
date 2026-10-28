import tensorflow as tf

def convertModel():

    try:
        # load tuned best .keras model
        model = tf.keras.models.load_model("tunedBestModel.keras")
        
        # convert to .tflite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        model_tflite = converter.convert()

        # save converted model
        with open("convertedModel.tflite", "wb") as f:
            f.write(model_tflite)

    except Exception as e:
        print(f"Failed to save the converted model. {e}")