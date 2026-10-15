import tensorflow as tf
from tensorflow.keras import layers, models
from imageDataLoad import loadImageData

# load image data
train_dataset, validation_dataset, class_names = loadImageData()

def buildModel():
    # import the MobileNetV3Large model from Keras applications
    base_model = tf.keras.applications.MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        #input_shape=(224, 224, 3)
    )

    # freeze the pretrained weights of the base model
    base_model.trainable = False

    # new classification head
    inputs = tf.keras.layers.Input(shape = (224, 224, 3))

    # keep the base model in inference mode
    x = base_model(inputs, training = False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = tf.keras.layers.Dense(
        1, 
        activation='sigmoid')
    (x)

    # assemble the complete model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model