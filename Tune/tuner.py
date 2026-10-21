import tensorflow as tf
import keras_tuner as kt
from Tune.tunableModel import buildModel
from model.imageDataLoad import loadImageData

# load image data
train_dataset, validation_dataset, class_names, class_weight = loadImageData()

def hypermodel(hp):

    # define hyperparameters to tune
    embedding_dim = hp.Int('embedding_dim', min_value=64, max_value=256, step=32)
    scale = hp.
    dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    augmentation_strength = hp.Float('augmentation_strength', min_value=0.5, max_value=2.0, step=0.1)

    # build the model
    classifierModel = buildModel(class_names, 
                                embedding_dim = embedding_dim,
                                scale = scale,
                                dropout = dropout,
                                augmentation_strength = augmentation_strength)

    # configure the model for training and evaluation
    classifierModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    # callbacks for early stopping
    callbacks = [
        # early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 5,
            restore_best_weights = True
        )

        # reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLRonPlateau(
            monitor = 'val_loss',
            factor = 0.2,
            patience = 2,
            min_lr = 1e-6
        )
    ]

    # training the model
    history = classifierModel.fit(
        train_dataset,
        validation_data = validation_dataset,
        epochs = 50,
        callbacks = callbacks,
        class_weight = class_weight       
    )

return classifiermodel, history