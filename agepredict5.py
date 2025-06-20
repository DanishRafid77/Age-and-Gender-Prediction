import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import os

# Load and preprocess dataset
def load_data(data_dir):
    images, ages, genders = [], [], []
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            # Parse file name
            parts = file.split('_')  # e.g., age_gender_other.jpg
            age, gender = int(parts[0]), int(parts[1])
            
            # Read and resize image
            img = cv2.imread(os.path.join(data_dir, file))
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            
            images.append(img)
            ages.append(age)
            genders.append(gender)
    
    images = np.array(images, dtype="float32") / 255.0  # Normalize
    ages = np.array(ages)
    genders = np.array(genders)
    return images, ages, genders

# Dataset directory (replace with your dataset path)
DATA_DIR = r"C:\Users\danis\OneDrive\Desktop\utkfacestuff\UTKFace"
images, ages, genders = load_data(DATA_DIR)

# Train-test split
x_train, x_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    images, ages, genders, test_size=0.2, random_state=42
)
y_gender_train = to_categorical(y_gender_train, 2)
y_gender_test = to_categorical(y_gender_test, 2)

# Model architecture
inputs = Input(shape=(128, 128, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)

# Age prediction output
age_output = Dense(1, activation='linear', name='age_output')(x)

# Gender prediction output
gender_output = Dense(2, activation='softmax', name='gender_output')(x)

# Define model
model = Model(inputs=inputs, outputs=[age_output, gender_output])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'age_output': 'mean_squared_error', 'gender_output': 'binary_crossentropy'},
    metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
)

# Train model
history = model.fit(
    x_train, {'age_output': y_age_train, 'gender_output': y_gender_train},
    validation_data=(x_test, {'age_output': y_age_test, 'gender_output': y_gender_test}),
    epochs=20,
    batch_size=32
)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Gender accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['gender_output_accuracy'], label='Gender Accuracy')
plt.plot(history.history['val_gender_output_accuracy'], label='Val Gender Accuracy')
plt.legend()
plt.title('Gender Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Total Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Total Loss')

plt.show()

# Save model
model.save("age_gender_model.h5")
