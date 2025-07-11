import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Configuration ===
CSV_PATH = 'yamnet_features.csv'
MODEL_PATH = 'models/distraction_classifier.h5'
LABEL_PATH = 'models/label_classes.npy'
CLASSES = ['baby_crying', 'music', 'animal_sound', 'yawning', 'talking']

print("📄 Loading CSV...")
df = pd.read_csv(CSV_PATH)

# Drop non-feature columns
if 'filename' in df.columns:
    df = df.drop(columns=['filename'])

X = df.drop(columns=['label']).values  # features
y = df['label'].values                 # target labels

print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features.")

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("🧠 Encoded classes:", list(label_encoder.classes_))

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")

# === Build Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Train ===
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    verbose=1
)

# === Evaluate ===
print("📈 Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2%}")

# === Save Model ===
print("💾 Saving model and labels...")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
np.save(LABEL_PATH, label_encoder.classes_)

print("🎉 Done! Model saved to:", MODEL_PATH)
