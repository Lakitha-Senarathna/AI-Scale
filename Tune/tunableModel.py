import tensorflow as tf
from tensorflow.keras import layers, models

def buildModel(class_names, embedding_dim=128, scale=16.0, dropout=0.2, augmentation_strength=1.0):

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

    # data augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),,
        tf.keras.layers.RandomRotation(0.05 * augmentation_strength),
        tf.keras.layers.RandomZoom(0.10 * augmentation_strength),
        tf.keras.layers.RandomTranslation(
            height_factor = 0.05 * augmentation_strength,
            width_factor = 0.05 * augmentation_strength
        ),
        tf.keras.layers.RandomContrast(0.10 * augmentation_strength),
        tf.keras.layers.RandomBrightness(factor=0.10 * augmentation_strength)
    ], name = "dataAugmentation")

    x = data_augmentation(inputs)
    x = base_model(x, training = False)     # keep the base model in inference mode

    # reduce the dimension of the feature maps to 1D vector
    x = tf.keras.layers.GlobalAveragePooling2D(name = "globalAveragePooling")(x)
    x = tf.keras.layers.Dropout(dropout, name = "dropout")(x)

    x = tf.keras.layers.Dense(embedding_dim, use_bias=False, name = "produceEmbedding")(x)
    x = tf.keras.layers.BatchNormalization(name = "batchNormalization")(x)
    
    # normalize the embeddings to unit length
    embeddings = tf.keras.layers.UnitNormalization(
        axis = -1, 
        name = "unitNormalization"
        )(x)

    # scale the embeddings so that the classification head can learn a better decision boundary
    scaleEmbeddings = tf.keras.layers.Rescaling(
        scale = scale, 
        name = "scaleEmbeddings"
        )(embeddings)
    
    # final classification head
    outputs = tf.keras.layers.Dense(
        len(class_names), 
        use_bias = False,
        kernel_constraint = tf.keras.constraints.UnitNorm(axis = 0),
        activation = 'softmax',
        name = "classificationHead"
        )(scaleEmbeddings)

    # model with final classification head
    classifier = tf.keras.Model(
        inputs = inputs,
        outputs = outputs,
        name = "produceClassifier"
        )

    return classifier