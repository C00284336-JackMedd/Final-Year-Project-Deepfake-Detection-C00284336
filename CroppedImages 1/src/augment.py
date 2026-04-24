import cv2
import os
from imgaug import augmenters as iaa

real_in = r"C:\Users\jackm\Downloads\CroppedImages\raw_images\real"
fake_in = r"C:\Users\jackm\Downloads\CroppedImages\raw_images\fake"

real_out = r"C:\Users\jackm\Downloads\CroppedImages\augmented\real"
fake_out = r"C:\Users\jackm\Downloads\CroppedImages\augmented\fake"

os.makedirs(real_out, exist_ok=True)
os.makedirs(fake_out, exist_ok=True)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Rotate((-30, 30)),
    iaa.GammaContrast((0.5, 2.0)),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.MultiplyBrightness((0.5, 1.5)),
    iaa.MultiplySaturation((0.5, 1.5)),
])

def augment_folder(input_folder, output_folder, n=4):
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        img = cv2.imread(os.path.join(input_folder, filename))

        for i in range(n):
            aug = seq(image=img)
            out_name = f"{filename.split('.')[0]}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, out_name), aug)

augment_folder(real_in, real_out)
augment_folder(fake_in, fake_out)
