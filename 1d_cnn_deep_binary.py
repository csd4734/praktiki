import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import my_utils

# Load the data
X_train, y_train, X_test, y_test = my_utils.read_data_to_binary('../train_classification.npz', '../test_inter_classification.npz')

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print(f"The length of the label encoder is {len(label_encoder.classes_)}")
print(f"The classses are {label_encoder.classes_}") 

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}
# One-hot encode the labels for multiclass classification
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f"The length of the label encoder is {len(label_encoder.classes_)}")
print(f"The length of the y_train is {len(y_train)}")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define a simpler 1D CNN model
model = Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),  # Input layer
    Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    # Dense(128, activation='relu', kernel_initializer='he_normal'),
    # Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile the model with a smaller learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the model checkpoint callback
checkpoint_callback = ModelCheckpoint('best_model.keras', save_best_only=True)

# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_val, y_val),
    class_weight=class_weights,  # Adjust based on class imbalance
    callbacks=[checkpoint_callback, lr_scheduler]
)

# # load the model
# model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
report = classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_)

print('Accuracy: ', accuracy)
print(report)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Calculate additional metrics: Sensitivity, Specificity, Average Score, Harmonic Score
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
average_score = (sensitivity + specificity) / 2
harmonic_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)

print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Average Score: {average_score}')
print(f'Harmonic Score: {harmonic_score}')
