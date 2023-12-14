# Importar librerias
import os
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Rutas de entrenamiento y validación
entrenamiento = 'C:/Users/elnik/source/repos/Proyecto reconocimiento de imagenes/Proyecto reconocimiento de imagenes/gatos_vs_perros/Entrenamiento'
validacion = 'C:/Users/elnik/source/repos/Proyecto reconocimiento de imagenes/Proyecto reconocimiento de imagenes/gatos_vs_perros/Validacion'

# Listas de imágenes
listaEntrenamiento = os.listdir(entrenamiento)
listaValidacion = os.listdir(validacion)

# Variables
ancho, alto = 100, 100

etiquetas = []
fotos = []
datos_train = []

etiquetas2 = []
fotos2 = []
datos_validacion = []

# Entrenamiento
for con, nameDir in enumerate(listaEntrenamiento):
    nombre = os.path.join(entrenamiento, nameDir)
    
    for nameFile in os.listdir(nombre):
        etiquetas.append(con)
        ruta_imagen = os.path.join(nombre, nameFile)
        
        try:
            foto = cv2.imread(ruta_imagen)
            
            if foto is None:
                raise Exception(f"Error al cargar la imagen: {ruta_imagen}")
            
            foto = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
            foto = cv2.resize(foto, (ancho, alto), interpolation=cv2.INTER_CUBIC)
            foto = foto.reshape(ancho, alto, 1)
            datos_train.append([foto, con])
            fotos.append(foto)
        except Exception as e:
            print(e)
            os.remove(ruta_imagen)

# Validación
for con2, nameDir in enumerate(listaValidacion):
    nombre = os.path.join(validacion, nameDir)
    
    for nameFile in os.listdir(nombre):
        etiquetas2.append(con2)
        ruta_imagen = os.path.join(nombre, nameFile)
        
        try:
            foto = cv2.imread(ruta_imagen)
            
            if foto is None:
                raise Exception(f"Error al cargar la imagen: {ruta_imagen}")
            
            foto = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
            foto = cv2.resize(foto, (alto, ancho), interpolation=cv2.INTER_CUBIC)
            foto = foto.reshape(alto, ancho, 1)
            datos_validacion.append([foto, con2])
            fotos2.append(foto)
        except Exception as e:
            print(e)
            os.remove(ruta_imagen)

# Normalización de imágenes
fotos = np.array(fotos).astype(np.float32) / 255.0
fotos2 = np.array(fotos2).astype(np.float32) / 255.0
etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)

# Verificar y ajustar la longitud de las etiquetas
min_length = min(len(etiquetas), len(etiquetas2))
etiquetas = etiquetas[:min_length]
etiquetas2 = etiquetas2[:min_length]
fotos = fotos[:min_length]
fotos2 = fotos2[:min_length]

# Dividir datos de validación para usar en model.fit
fotos, fotos_val, etiquetas, etiquetas_val = train_test_split(fotos, etiquetas, test_size=0.2, random_state=42)

# Generador de imágenes tf
imgTrainGen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.5, 1.5],
    vertical_flip=True,
    horizontal_flip=True
)

imgTrainGen.fit(fotos)

# Modelo con Capas convolucionales y Dropout
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ancho, alto, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#longitudes de los conjuntos de datos
print("Longitud de datos de entrenamiento:", len(fotos), len(etiquetas))
print("Longitud de datos de validacion:", len(fotos_val), len(etiquetas_val))

# Compilación del modelo
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Entrenamiento del modelo
BoardCNN = TensorBoard(log_dir='C:/Users/elnik/source/repos/Proyecto reconocimiento de imagenes/Proyecto reconocimiento de imagenes')
modelo.fit(imgTrainGen.flow(fotos, etiquetas, batch_size=32),
           validation_data=(fotos_val, etiquetas_val),
           epochs=100, callbacks=[BoardCNN])


# Guardar modelo
modelo.save('ModeloCNN.h5')
modelo.save_weights('pesosCNN.h5')
print("Modelo guardado")
