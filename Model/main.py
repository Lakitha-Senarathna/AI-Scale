from Model.evaluate import evaluateModel
from Model.imageDataLoad import loadImageData
from Model.train import trainModel
from Tune.tuner import tuneModel

train_dataset, validation_dataset, class_names, class_weight = loadImageData()

# train baseline model
#classifierModel, history = trainModel()

# tuned model
best_tuned_model = tuneModel(train_dataset, validation_dataset, class_weight)

evaluateModel(best_tuned_model, validation_dataset, class_names)
