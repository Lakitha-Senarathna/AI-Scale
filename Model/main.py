from model.evaluate import evaluateModel
from model.imageDataLoad import loadImageData
from model.train import trainModel
from tune.tuner import tuneModel

train_dataset, validation_dataset, class_names, class_weight = loadImageData()

# train baseline model
#classifierModel, history = trainModel()

# tuned model
best_tuned_model = tuneModel(train_dataset, validation_dataset, class_weight)

evaluateModel(best_tuned_model, validation_dataset, class_names)
