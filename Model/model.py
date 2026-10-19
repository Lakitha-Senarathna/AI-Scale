import tensorflow as tf
from tensorflow.keras import layers, models
from imageDataLoad import loadImageData

# load image data
train_dataset, validation_dataset, class_names = loadImageData()

def buildModel(class_names):
    # import the MobileNetV3Large model from Keras applications
    base_model = tf.keras.applications.MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        include_preprocessing=True,
        input_shape=(224, 224, 3)
    )

    # freeze the pretrained weights of the base model
    base_model.trainable = False

    # new classification head
    inputs = tf.keras.layers.Input(shape = (224, 224, 3), name = "imageInput")

    # data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
        tf.keras.layers.RandomContrast(0.10),
        tf.keras.layers.RandomBrightness(factor=0.10)
    ], name = "dataAugmentation")

    x = data_augmentation(inputs)
    x = base_model(x, training = False)     # keep the base model in inference mode

    # reduce the dimension of the feature maps to 1D vector
    x = tf.keras.layers.GlobalAveragePooling2D(name = "globalAveragePooling")(x)
    x = tf.keras.layers.Dropout(0.2, name = "dropout")(x)

    x = tf.keras.layers.Dense(128, use_bias=False, name = "produceEmbedding")(x)
    x = tf.keras.layers.BatchNormalization(name = "batchNormalization")(x)
    
    embeddings = tf.keras.layers.UnitNormalization(axis = -1, name = "unitNormalization")(x)
    
    outputs = tf.keras.layers.Dense(
        len(class_names), 
        activation='softmax',
        name = "classificationHead"
        )(embeddings)

    # assemble the complete model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model