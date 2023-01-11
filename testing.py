from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_Model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r').readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open(r'C:\Users\Acer\Desktop\7.png').convert('RGB')

image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

print('Class:', class_name, end='')
print('Confidence score:', confidence_score)