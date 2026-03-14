# train_nn.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import collections

#Getting 99 - 100 % accuracy
# Load data
all_letters_landmarks = np.load("arabic_letters_handmarks.npy", allow_pickle=True).item()

X, y = [], []

for letter, images in all_letters_landmarks.items():
    for img in images:
        X.append(img.flatten())  # 21*3 = 63 features
        y.append(letter)

X = np.array(X)
y = np.array(y)

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.99, random_state=42, stratify=y_encoded)




# Count train/test samples per label
train_counts = collections.Counter(y_train)
test_counts = collections.Counter(y_test)

for i, letter in enumerate(le.classes_):
    print(f"{letter}: train={train_counts[i]}, test={test_counts[i]}")

# Build the NN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# Print train/val accuracy from history
print("Train acc:", history.history['accuracy'][-1])
print("Val acc:", history.history['val_accuracy'][-1])
# Save model and encoder
model.save("hand_landmarks_nn_model.keras")
np.save("label_encoder.npy", le.classes_)