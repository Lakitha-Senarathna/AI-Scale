import tensorflow as tf
import keras_tuner as kt
from Model.model import buildModel

def getHyperparameters(class_names):

    def hyperparameters(hp):

        # define hyperparameters to tune
        embedding_dim = hp.Int(
            'embedding_dim',
            min_value = 64,
            max_value = 512,
            step = 32)

        scale = hp.Float(
            'scale',
            min_value = 8.0,
            max_value = 32.0,
            step = 4.0)

        dropout = hp.Float(
            'dropout',
            min_value = 0.1,
            max_value = 0.5,
            step = 0.1)

        augmentation_strength = hp.Float(
            'augmentation_strength',
            min_value = 0.5,
            max_value = 2.0,
            step = 0.5)

        learning_rate = hp.Float(
            'learning_rate', 
            min_value = 1e-4, 
            max_value = 1e-3,
            sampling = 'log')

        # build the model
        classifierModel = buildModel(
            class_names, 
            embedding_dim = embedding_dim,
            scale = scale,
            dropout = dropout,
            augmentation_strength = augmentation_strength
        )

        # define loss for model compilation
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # configure the model for training and evaluation
        classifierModel.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
            loss = loss,
            metrics = ['accuracy']
        )

        return classifierModel
    
    return hyperparameters

def tuneModel(train_data, validation_data, class_weight, class_names):

    # define the tuner
    tuner = kt.Hyperband(
        hypermodel = getHyperparameters(class_names),
        objective = 'val_accuracy',
        max_epochs = 100,
        seed = 42,
        factor = 3,
        directory = 'tuner_results',
        project_name = 'image_classification_tuning',
        executions_per_trial = 3
    )

    def callbacks():
        # callbacks for early stopping and learning rate adjustment
        callbacks = [
            # early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor = 'val_accuracy',
                mode = 'max',
                patience = 5,
                restore_best_weights = True
            ),

            # reduce learning rate when validation loss plateaus
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = 'val_accuracy',
                mode = 'max',
                factor = 0.2,
                patience = 2,
                min_lr = 1e-6
                )
        ]

        return callbacks

    tuner.search(
        train_data,
        validation_data = validation_data,
        class_weight = class_weight,
        callbacks = callbacks()
    )

    # retrieve best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

    # build the final model with best hyperparameters
    best_model = tuner.hypermodel.build(best_hyperparameters)

    # train the final parameters
    history = best_model.fit(
        train_data,
        validation_data = validation_data,
        epochs = 100,
        class_weight = class_weight,
        callbacks = callbacks()
    )

    try:
        # save best model
        best_model.save("tunedBestModel.keras")
        print(f"Tuned Model Saved Successfully. Best hyperparameters {best_hyperparameters.values}")
        
    except Exception as e:
        print(f"Failed to save the model 'tunedBestModel.keras' {e}")

    return best_model
