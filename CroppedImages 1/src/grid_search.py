import os
import csv
import itertools
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from mesonet import Meso4

# ============================
# PATHS
# ============================

train_dir = r"C:\Users\jackm\Downloads\CroppedImages\dataset_folder\train"
val_dir   = r"C:\Users\jackm\Downloads\CroppedImages\dataset_folder\validation"
test_dir  = r"C:\Users\jackm\Downloads\CroppedImages\dataset_folder\test"

results_csv = r"C:\Users\jackm\Downloads\CroppedImages\grid_search_results.csv"

# ============================
# HYPERPARAMETERS TO SEARCH
# ============================

learning_rates = [0.001, 0.0005]
batch_sizes    = [16, 32]
epochs_list    = [10, 20]

# ============================
# IMAGE GENERATORS
# ============================

def make_train_gen(batch_size):
    return ImageDataGenerator(
        rescale=1/255,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True
    ).flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="binary"
    )

def make_val_gen(batch_size):
    return ImageDataGenerator(
        rescale=1/255
    ).flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="binary"
    )

def make_test_gen():
    return ImageDataGenerator(
        rescale=1/255
    ).flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=1,
        class_mode="binary",
        shuffle=False
    )

# ============================
# GRID SEARCH LOOP
# ============================

# Create CSV file if not exists
if not os.path.exists(results_csv):
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "learning_rate",
            "batch_size",
            "epochs",
            "val_accuracy",
            "test_accuracy"
        ])

# Iterate through all combinations
for lr, batch, epochs in itertools.product(learning_rates, batch_sizes, epochs_list):

    print("\n====================================")
    print(f"Training model with:")
    print(f"LR={lr}, Batch={batch}, Epochs={epochs}")
    print("====================================\n")

    # Generators
    train_gen = make_train_gen(batch)
    val_gen   = make_val_gen(batch)
    test_gen  = make_test_gen()

    # Build model
    model = Meso4(lr=lr).model

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]

    # Train
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Validation accuracy
    val_acc = max(history.history["val_accuracy"])

    # Test accuracy
    loss, test_acc = model.evaluate(test_gen, verbose=1)

    # Save results
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([lr, batch, epochs, val_acc, test_acc])

    print(f"\nFinished: LR={lr}, Batch={batch}, Epochs={epochs}")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")
    print("====================================\n")
