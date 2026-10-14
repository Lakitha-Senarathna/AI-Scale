import tensorflow as tf

model = tf.keras.applications.MobileNetV3Large(
    weights='imagenet',
    include_top=False, 
    input_shape=(224, 224, 3)
    weights='imagenet',)