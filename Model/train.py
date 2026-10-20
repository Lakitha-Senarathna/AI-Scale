import tensorflow as tf
from model.model import buildModel
from model.imageDataLoad import loadImageData

def trainModel():
    # load image data
    train_dataset, validation_dataset, class_names, class_weight = loadImageData()

    # build the model
    classifierModel, featureExtractor = buildModel(class_names)

    # configure the model for training and evaluation
    classifierModel.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    # callbacks for early stopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 5,
            restore_best_weights = True
        )

        tf.keras.callbascks.ModelCheckpoint(
            filepath = "bestClassifier.keras",
            monitor = 'val_loss',
            save_best_only = True
        )

        tf.keras.callbacks.ReduceLRonPlateau(
            monitor = 'val_loss',
            factor = 0.2,
            patience = 2
        )
    ]

    # training the model
    classifierModel.fit(
        train_dataset,
        validation_data = validation_dataset,
        epochs = 50,
        callbacks = callbacks,
        class_weight = class_weight       
    )

