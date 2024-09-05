# Importar librerias

import tensorflow as tf
import cv2
import numpy as np
from keras_preprocessing.image import img_to_array

# Ruta de Modelo

modelo = 'Your_Model_Directory'

# Lectura de redes neuronales

cnn = tf.keras.models.load_model(modelo)
pesosCnn = cnn.get_weights()
cnn.set_weights(pesosCnn)

# VideoCapura

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_CUBIC)
    
    #Normalizacion de imagen
    gray = np.array(gray).astype('float32') / 255

    #Convercion de imagen a matriz

    img = img_to_array(gray)
    img = np.expand_dims(img, axis=0)

    # Prediccion de imagen
    prediccion = cnn.predict(img)
    prediccion = prediccion[0]
    prediccion = prediccion[0]
    print(prediccion)

    # Clasificacion de imagen
    if prediccion <= 0.5:
        cv2.putText(frame, 'Gato', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'Perro', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

   # Muestreo de fotogramas
   cv2.imshow("CNN", frame)


   t = cv2.waitKey(1)
   if t == 27:
       break
cv2.destroyAllWindows()
cap.release()
