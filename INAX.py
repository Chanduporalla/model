import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Image Data Generator with Augmentation for Training and Validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'lung-data/Data/train/',
    target_size=(224, 224),  # Updated to match InceptionV3 input size
    batch_size=32,
    class_mode='categorical'  # Multiclass classification
)

validation_generator = validation_datagen.flow_from_directory(
    'lung-data/Data/valid/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Multiclass classification
)

# Load InceptionV3 with pre-trained weights
inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Feature extraction layers
x = inception_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 4 classes: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal

# Define the model
model = Model(inputs=inception_base.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=25
)

# Evaluate the Model
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {test_acc}")

# Extract features using InceptionV3 model
features_train = model.predict(train_generator)
features_valid = model.predict(validation_generator)

# Train a multiclass SVM
svm_clf = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovr'))
svm_clf.fit(features_train, train_generator.classes)

# Predict using the SVM classifier
y_pred = svm_clf.predict(features_valid)

# Confusion Matrix and Classification Report
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
target_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma', 'Normal']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# Plot Training & Validation Accuracy and Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

