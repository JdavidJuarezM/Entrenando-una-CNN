from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Directorio de entrenamiento
train_dir = 'D:\Octavo Semestre\Vision Artificial\Actividad. Entrenando una CNN'

# Generador de imágenes para ajuste y aumento del conjunto de datos 
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,  # Mejora el aumento de datos
    brightness_range=(0.8, 1.2),
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',  # Asegúrate de usar 'categorical'
    subset='training')

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',  # Asegúrate de usar 'categorical'
    subset='validation')

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases: amor_paz, aceptacion, declinacion
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Guardar el modelo y los pesos usando el formato sugerido
model.save('ModeloS.keras')  # Guardar el modelo usando el nuevo formato Keras
model.save_weights('pesosS.weights.h5')  # Guardar los pesos con la extensión correcta