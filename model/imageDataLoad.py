import tensorflow as tf
from pathlib import Path

# image directory
DATA_DIRECTORY = Path("")

# configuration parameters for loading image data
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
SEED  = 123
VALIDATION_SPLIT = 0.2

# load image data from directory and split into training and validation datasets
def loadImageData():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    validation_split = VALIDATION_SPLIT,
    seed = SEED,
    subset = "training",
    shuffle = True,
    batch_size = BATCH_SIZE,
    image_size = IMAGE_SIZE 
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    validation_split = VALIDATION_SPLIT,
    seed = SEED,
    subset = "validation",
    shuffle = True,
    batch_size = BATCH_SIZE,
    image_size = IMAGE_SIZE
    )

    class_names = train_dataset.class_names

    # optimize the datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size = AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size = AUTOTUNE)

    return train_dataset, validation_dataset, class_names