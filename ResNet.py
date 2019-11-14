from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

resNet_model = ResNet50(weights='imagenet')
img = image.load_img("police_van.jpeg", target_size=(224, 224))
input = image.img_to_array(img)
input = np.expand_dims(input, axis=0)
input = preprocess_input(input)

predictions = resNet_model.predict(input)
decoded_predictions = decode_predictions(predictions, top=3)[0]
print('Predicted classes:')
print(decoded_predictions[0])
print(decoded_predictions[1])
print(decoded_predictions[2])
