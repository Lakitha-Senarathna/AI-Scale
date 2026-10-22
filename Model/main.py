from model.train import trainModel
from model.evaluate import evaluateModel

classifierModel, history = trainModel()
evaluateModel()
