from model.model import buildModel

baseModel, featureExtractor = buildModel()

featureExtractor.save("featureExtractor.keras")
baseModel.save("produceClassifier.keras")