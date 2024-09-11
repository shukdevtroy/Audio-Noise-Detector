import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
from keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Load audio file
filename = str(input('Enter path to file: '))
sr, y = wavfile.read(filename)

features = np.random.rand(40, 5)

# Reshape the features as needed by the model
x_test = np.reshape(features, (1, features.shape[0], features.shape[1], 1))

# Make predictions using the model
ans = model.predict(x_test)

# Define class labels
class_labels = {
    0: 'Windy',
    1: 'Horn',
    2: 'Children-noise',
    3: 'Dog Bark',
    4: 'Drilling',
    5: 'Engine Idling',
    6: 'Gun Shot',
    7: 'Jackhammer',
    8: 'Siren',
    9: 'Street music'
}

# Process the predictions to get the resulting indices
indices = np.argsort(ans[0])[::-1][:3]  # Getting the top 3 predicted classes

# Print the predicted classes
print('Noises Present:')
for idx in indices:
    print(class_labels[idx])