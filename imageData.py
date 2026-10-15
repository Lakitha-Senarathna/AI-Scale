import tensorflow as tf
import os
import pathlib

dataset_URL = ""

archive_path = tf.keras.utils.get_file(
    origin=dataset_URL, 
    fname="flower_photos.tgz", 
    extract=True
    )

data_dir = pathlib.Path(archive_path).parent / "flower_photos"

batch_size = 32
image_size = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset = "training",
    shuffle = True,
    batch_size = batch_size,
    image_size = image_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset = "validation",
    shuffle = True,
    batch_size = batch_size,
    image_size = image_size
)