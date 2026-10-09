from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam

from config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    CLASS_NAMES,
    NUM_CLASSES,
    MODEL_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    LR_REDUCTION_PATIENCE,
    LR_REDUCTION_FACTOR,
    MIN_LEARNING_RATE,
)

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Loading datasets...")
print("=" * 60)

train = np.load(TRAIN_DATA_PATH)
validation = np.load(VALIDATION_DATA_PATH)
test = np.load(TEST_DATA_PATH)

X_train = train["X"]
y_train = train["y"]

X_validation = validation["X"]
y_validation = validation["y"]

X_test = test["X"]
y_test = test["y"]

print(f"Training:   {X_train.shape}")
print(f"Validation: {X_validation.shape}")
print(f"Test:       {X_test.shape}")

TIME_STEPS = X_train.shape[1]
NUM_FEATURES = X_train.shape[2]

print(f"\nTime steps : {TIME_STEPS}")
print(f"Features   : {NUM_FEATURES}")
print(f"Classes    : {NUM_CLASSES}")

print("\nComputing class weights...")

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)

class_weights = dict(enumerate(weights))

print(class_weights)

class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = Dense(64, activation="tanh")
        self.score = Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        weights = tf.nn.softmax(self.score(x), axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

model = Sequential([
    Input(shape=(TIME_STEPS, NUM_FEATURES)),

    Bidirectional(LSTM(128, return_sequences=True)),
    LayerNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True)),
    LayerNormalization(),
    Dropout(0.3),

    AttentionPooling(),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

#############################################################
# CALLBACKS
#############################################################

best_model_path = MODEL_DIR / "best_csi_bilstm_attention.keras"

callbacks = [

    ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_REDUCTION_FACTOR,
        patience=LR_REDUCTION_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
    ),
]

#############################################################
# TRAIN
#############################################################

print("\n" + "=" * 60)
print("Training...")
print("=" * 60)

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_validation, y_validation),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1,

)

#############################################################
# LOAD BEST MODEL
#############################################################

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    best_model_path,
    custom_objects={"AttentionPooling": AttentionPooling}
)

#############################################################
# TEST
#############################################################

print("\n" + "=" * 60)
print("Testing...")
print("=" * 60)

loss, accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=0,
)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

#############################################################
# PREDICTIONS
#############################################################

predictions = best_model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

#############################################################
# CLASSIFICATION REPORT
#############################################################

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        predicted_labels,

        target_names=CLASS_NAMES,

        digits=4,

    )

)

#############################################################
# CONFUSION MATRIX
#############################################################

cm = confusion_matrix(
    y_test,
    predicted_labels,
)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=CLASS_NAMES,

)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45,
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
)

plt.show()

#############################################################
# TRAINING HISTORY
#############################################################

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy.png",
    dpi=300,
)

plt.show()

#############################################################
# LOSS
#############################################################

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss.png",
    dpi=300,
)

plt.show()

#############################################################
# SAVE HISTORY
#############################################################

history_file = RESULTS_DIR / "lstm_history.csv"

np.savetxt(

    history_file,

    np.column_stack([

        history.history["accuracy"],

        history.history["val_accuracy"],

        history.history["loss"],

        history.history["val_loss"],

    ]),

    delimiter=",",

    header="accuracy,val_accuracy,loss,val_loss",

    comments="",

)

#############################################################
# FINISHED
#############################################################

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(f"\nModel saved to:")
print(best_model_path)

print("\nResults saved to:")
print(RESULTS_DIR)