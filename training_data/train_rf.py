# train_rf.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#alomst same results 99 - 100 % accuracy
# Load data
all_letters_landmarks = np.load("arabic_letters_handmarks.npy", allow_pickle=True).item()

X, y = [], []
for letter, images in all_letters_landmarks.items():
    for img in images:
        X.append(img.flatten())
        y.append(letter)

X = np.array(X)
y = np.array(y)

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest accuracy: {acc * 100:.4f}")

# Save model and encoder
import joblib
joblib.dump(rf, "hand_landmarks_rf_model.pkl")
np.save("label_encoder.npy", le.classes_)