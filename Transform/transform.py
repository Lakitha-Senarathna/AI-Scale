import tensorflow as tf

# load tuned best .keras model
model = tf.keras.models.load_model("tunedBestModel.keras")

# convert to .tflite format
model_tflite = tf.lite.TFLiteConverter.from_keras_model(model)