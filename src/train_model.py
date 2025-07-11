import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

print("Loading libraries...")

# Configuration
DATASET_PATH = 'dataset'
SAMPLE_RATE = 16000
CLASSES = ['baby_crying', 'music', 'animal_sound', 'yawning', 'talking']

# Step 1: Load audio files
print("\nStep 1: Loading audio files...")
audio_files = []
labels = []

for class_name in CLASSES:
    class_dir = os.path.join(DATASET_PATH, class_name)
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3'))]
        print(f"Found {len(files)} files in {class_name}")
        
        for file in files:
            audio_files.append(os.path.join(class_dir, file))
            labels.append(class_name)

print(f"\nTotal files: {len(audio_files)}")

if len(audio_files) == 0:
    print("ERROR: No audio files found! Please add audio files to dataset folders.")
    exit()

# Step 2: Load YAMNet
print("\nStep 2: Loading YAMNet model (this may take a minute)...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet loaded!")

# Step 3: Extract features
print("\nStep 3: Extracting features from audio files...")
features = []
valid_labels = []

for i, (file_path, label) in enumerate(zip(audio_files, labels)):
    print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
    
    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # Get YAMNet embeddings
        scores, embeddings, spectrogram = yamnet_model(audio)
        
        # Average the embeddings
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        features.append(avg_embedding)
        valid_labels.append(label)
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Convert to numpy arrays
X = np.array(features)
y = np.array(valid_labels)

# Encode labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFeatures shape: {X.shape}")
print(f"Classes: {label_encoder.classes_}")

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 5: Load existing model
print("\nStep 5: Loading existing model...")
model = load_model('models/distraction_classifier.h5')
print("Model loaded!")

# Optional: recompile if needed (e.g., change learning rate)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# Step 6: Train model for more epochs
print("\nStep 6: Continuing training...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,  # Set how many more epochs you want
    batch_size=16,
    verbose=1
)

# Step 7: Evaluate
print("\nStep 7: Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.2%}")

# Step 8: Save model
print("\nStep 8: Saving updated model...")
os.makedirs('models', exist_ok=True)
model.save('models/distraction_classifier.h5')
np.save('models/label_classes.npy', label_encoder.classes_)

print("\n✅ Training complete! Model updated and saved to 'models/distraction_classifier.h5'")




# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import os
# import librosa
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# print("Loading libraries...")

# # Configuration
# DATASET_PATH = 'dataset'
# SAMPLE_RATE = 16000
# CLASSES = ['baby_crying', 'music', 'animal_sound', 'yawning', 'talking']

# # Step 1: Load audio files
# print("\nStep 1: Loading audio files...")
# audio_files = []
# labels = []

# for class_name in CLASSES:
#     class_dir = os.path.join(DATASET_PATH, class_name)
#     if os.path.exists(class_dir):
#         files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3'))]
#         print(f"Found {len(files)} files in {class_name}")
        
#         for file in files:
#             audio_files.append(os.path.join(class_dir, file))
#             labels.append(class_name)

# print(f"\nTotal files: {len(audio_files)}")

# if len(audio_files) == 0:
#     print("ERROR: No audio files found! Please add audio files to dataset folders.")
#     exit()

# # Step 2: Load YAMNet
# print("\nStep 2: Loading YAMNet model (this may take a minute)...")
# yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
# print("YAMNet loaded!")

# # Step 3: Extract features
# print("\nStep 3: Extracting features from audio files...")
# features = []
# valid_labels = []

# for i, (file_path, label) in enumerate(zip(audio_files, labels)):
#     print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
    
#     try:
#         # Load audio
#         audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
#         # Get YAMNet embeddings
#         scores, embeddings, spectrogram = yamnet_model(audio)
        
#         # Average the embeddings
#         avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
#         features.append(avg_embedding)
#         valid_labels.append(label)
        
#     except Exception as e:
#         print(f"  Error: {e}")
#         continue

# # Convert to numpy arrays
# X = np.array(features)
# y = np.array(valid_labels)

# # Encode labels to numbers
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# print(f"\nFeatures shape: {X.shape}")
# print(f"Classes: {label_encoder.classes_}")

# # Step 4: Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42
# )

# print(f"\nTraining samples: {len(X_train)}")
# print(f"Testing samples: {len(X_test)}")

# # Step 5: Build classifier
# print("\nStep 5: Building classifier model...")
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(1024,)),  # YAMNet outputs 1024 features
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(len(CLASSES), activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Step 6: Train model
# print("\nStep 6: Training model...")
# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=16,
#     verbose=1
# )

# # Step 7: Evaluate
# print("\nStep 7: Evaluating model...")
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"\nTest Accuracy: {test_accuracy:.2%}")

# # Step 8: Save model
# print("\nStep 8: Saving model...")
# os.makedirs('models', exist_ok=True)
# model.save('models/distraction_classifier.h5')
# np.save('models/label_classes.npy', label_encoder.classes_)

# print("\n✅ Training complete! Model saved to 'models/distraction_classifier.h5'")