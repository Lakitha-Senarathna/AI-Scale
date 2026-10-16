import tensorflow as tf
from model.model import buildModel
from model.imageDataLoad import loadImageData

# load image data
train_dataset, validation_dataset, class_names = loadImageData()

# build the model
model = buildModel()

def trainModel():

    # configure the model for training and evaluation
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    # training the model
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs = 10
    )