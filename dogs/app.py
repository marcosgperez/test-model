import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np

datos, metadatos = tfds.load(
    'cats_vs_dogs', as_supervised=True, with_info=True)

tfds.as_dataframe(datos['train'].take(5), metadatos)

plt.figure(figsize=(20, 20))

size = 100
datos_entrenamiento = []
X = []  # Lista de imagenes
y = []  # Lista de etiquetas

for i, (imagen, etiqueta) in enumerate(datos['train']):
    imagen = cv2.resize(imagen.numpy(), (size, size))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(size, size, 1)
    datos_entrenamiento.append([imagen, etiqueta])

for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen)
    y.append(etiqueta)

X = np.array(X).astype('float') / 255
y = np.array(y)

modeloDenso = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(size, size, 1)),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(size, size, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

modeloCNN2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(size, size, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

modeloDenso.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modeloCNN.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
modeloCNN2.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tensorboardCNN2 = tf.keras.callbacks.TensorBoard(log_dir='logs/cnn2')
modeloCNN2.fit(X, y, batch_size=32, validation_split=0.15,
               epochs=100, callbacks=[tensorboardCNN2])

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.7, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=15,
)

datagen.fit(X)

modeloDenso_ad = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(size, size, 1)),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN_ad = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(size, size, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

modeloCNN2_ad = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(size, size, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

modeloDenso_ad.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modeloCNN_ad.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modeloCNN2_ad.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_entrenamiento = X[:19700]
X_validacion = X[19700:]

y_entrenamiento = y[:19700]
y_validacion = y[19700:]

data_gen_entrenamiento = datagen.flow(
    X_entrenamiento, y_entrenamiento, batch_size=32)

tensorboardCNN2_ad = tf.keras.callbacks.TensorBoard(log_dir='logs/cnn2_ad')

# modeloCNN2_ad.fit(data_gen_entrenamiento, batch_size=32, epochs=100, validation_data=(
#     X_validacion, y_validacion), callbacks=[tensorboardCNN2_ad])

