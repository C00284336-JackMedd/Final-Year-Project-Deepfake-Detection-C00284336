from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mesonet import Meso4

test_dir = r"C:\Users\jackm\Downloads\CroppedImages\dataset_folder\test"
weights_path = r"C:\Users\jackm\Downloads\CroppedImages\best_weights.h5"

test_gen = ImageDataGenerator(rescale=1/255).flow_from_directory(
    test_dir, target_size=(128,128), batch_size=1, class_mode="binary", shuffle=False
)

meso = Meso4()
meso.model.load_weights(weights_path)

loss, acc = meso.model.evaluate(test_gen)
print("Test accuracy:", acc)
