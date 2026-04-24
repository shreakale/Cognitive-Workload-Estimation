import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


df = pd.read_csv('data/emotions.csv')
print(f" Loaded! Shape: {df.shape}")

label_map = {
    'POSITIVE' : 'Relaxed',
    'NEUTRAL'  : 'Focused',
    'NEGATIVE' : 'Confused'
}
df['cognitive_state'] = df['label'].map(label_map)
print("\n  Label Mapping:")
print(df['cognitive_state'].value_counts())


print("\n Selecting features...")
mean_cols = [c for c in df.columns if c.startswith('mean_')]
fft_cols  = [c for c in df.columns if c.startswith('fft_')]
feature_cols = mean_cols + fft_cols

X = df[feature_cols].values.astype(np.float32)
y = df['cognitive_state'].values

X = np.nan_to_num(X, nan=0.0)
print(f"   Total features: {X.shape[1]}")

print("\n Scaling + PCA...")
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca     = PCA(n_components=100, random_state=42)
X_pca   = pca.fit_transform(X_scaled)

explained = sum(pca.explained_variance_ratio_) * 100
print(f"   Reduced to 100 components")
print(f"   Variance explained: {explained:.1f}%")


le        = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat     = to_categorical(y_encoded, num_classes=3)
print(f"\n  Classes: {le.classes_}")


X_train, X_test, y_train, y_test, ye_train, ye_test = train_test_split(
    X_pca, y_cat, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\n Train: {X_train.shape[0]} samples")
print(f" Test : {X_test.shape[0]} samples")


print("\n  Building MLP model...")

model = Sequential([
    #Input Block
    Dense(512, activation='relu', input_shape=(100,)),
    BatchNormalization(),
    Dropout(0.4),

    #Hidden Block 1
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # Hidden Block 2
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    #Hidden Block 3
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    #Hidden Block 4
    Dense(32, activation='relu'),
    Dropout(0.2),

    #Output
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


print("\n Training MLP model...")

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


print("\n Evaluating model...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test Accuracy: {acc*100:.2f}%")
print(f" Test Loss    : {loss:.4f}")

y_pred     = model.predict(X_test)
y_pred_cls = np.argmax(y_pred, axis=1)

print("\n Classification Report:")
print(classification_report(
    ye_test, y_pred_cls,
    target_names=le.classes_
))


os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)


cm = confusion_matrix(ye_test, y_pred_cls)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')
plt.title(f'Confusion Matrix — MLP ({acc*100:.2f}%)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('output/confusion_matrix_mlp.png')
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

ax1.plot(history.history['accuracy'],
         label='Train Accuracy', color='blue')
ax1.plot(history.history['val_accuracy'],
         label='Val Accuracy',   color='orange')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],
         label='Train Loss', color='blue')
ax2.plot(history.history['val_loss'],
         label='Val Loss',   color='orange')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('output/training_history_mlp.png')
plt.show()

model.save('models/mlp_cognitive_load.keras')

print(f"Accuracy : {acc*100:.2f}%")
print(f"   Outputs  : output/ folder")
print("=" * 60)