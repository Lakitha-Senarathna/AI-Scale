from model.model import buildModel

model, featureExtractor = buildModel()

featureExtractor.save("featureExtractor.keras")
model.save("produceClassifier.keras")