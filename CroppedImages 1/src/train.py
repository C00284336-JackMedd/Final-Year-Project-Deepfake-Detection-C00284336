from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from mesonet import Meso4

train_dir = r"C:\Users\jackm\Downloads\CroppedImages\dataset_folder\train"
val_dir = r"C:\Users\jackm\Downloads\CroppedImages\dataset_folder\validation"
weights_path = r"C:\Users\jackm\Downloads\CroppedImages\best_weights.h5"

train_gen = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True
).flow_from_directory(
    train_dir, target_size=(128,128), batch_size=32, class_mode="binary"
)

val_gen = ImageDataGenerator(rescale=1/255).flow_from_directory(
    val_dir, target_size=(128,128), batch_size=32, class_mode="binary"
)

meso = Meso4()

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True)
]

meso.model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    callbacks=callbacks
)
