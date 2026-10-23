import tensorflow as tf
from model.model import buildModel
from model.imageDataLoad import loadImageData

def trainModel():
    # load image data
    train_dataset, validation_dataset, class_names, class_weight = loadImageData()

    embedding_dim = 128
    scale = 16.0
    dropout = 0.2
    augmentation_strength = 1  

    # build the model
    classifierModel = buildModel(class_names, embedding_dim, scale, dropout, augmentation_strength)

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
        ),

        # save the best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath = "bestClassifier.keras",
            monitor = 'val_loss',
            save_best_only = True
        ),

        # reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
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
        epochs = 15,
        callbacks = callbacks,
        class_weight = class_weight       
    )

    return classifierModel, history