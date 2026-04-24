import os
import shutil
import random

# Input folders (augmented dataset)
real_in = r"C:\Users\jackm\Downloads\CroppedImages\augmented\real"
fake_in = r"C:\Users\jackm\Downloads\CroppedImages\augmented\fake"

# Output base folder
base_out = r"C:\Users\jackm\Downloads\CroppedImages\dataset_folder"

splits = ["train", "validation", "test"]

# Create folder structure
for split in splits:
    os.makedirs(os.path.join(base_out, split, "real"), exist_ok=True)
    os.makedirs(os.path.join(base_out, split, "fake"), exist_ok=True)

def split_and_copy(input_folder, output_base):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))]
    random.shuffle(files)

    total = len(files)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # Copy files
    for f in train_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(output_base, "train"))

    for f in val_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(output_base, "validation"))

    for f in test_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(output_base, "test"))

    return len(train_files), len(val_files), len(test_files)

# Process REAL images
real_train, real_val, real_test = split_and_copy(
    real_in,
    os.path.join(base_out, "")
)

# Move them into correct subfolders
for split in splits:
    src = os.path.join(base_out, split)
    dst = os.path.join(base_out, split, "real")
    for f in os.listdir(src):
        if f.lower().endswith((".jpg", ".png")):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))

# Process FAKE images
fake_train, fake_val, fake_test = split_and_copy(
    fake_in,
    os.path.join(base_out, "")
)

# Move them into correct subfolders
for split in splits:
    src = os.path.join(base_out, split)
    dst = os.path.join(base_out, split, "fake")
    for f in os.listdir(src):
        if f.lower().endswith((".jpg", ".png")):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))

print("Done!")
print(f"REAL → Train: {real_train}, Val: {real_val}, Test: {real_test}")
print(f"FAKE → Train: {fake_train}, Val: {fake_val}, Test: {fake_test}")
