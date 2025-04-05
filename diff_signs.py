import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img

# ---------------------
# Settings
# ---------------------
NUM_CLASSES = 43
IMG_WIDTH, IMG_HEIGHT = 32, 32
dataset_dir = Path(r"C:\Users\1\Desktop\Dyne\Project\gtsrb-german-traffic-sign\versions\1\Train")  

# ---------------------
# Plot one sample per class
# ---------------------
plt.figure(figsize=(12, 8))
for i in range(NUM_CLASSES):
    class_dir = dataset_dir / str(i)
    images = list(class_dir.glob("*"))
    if not images:
        print(f"No images found for class {i}")
        continue

    img_path = images[0]
    img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

    plt.subplot(7, 7, i + 1)
    plt.imshow(img)
    plt.title(f"Class {i}")
    plt.axis('off')

plt.tight_layout()
plt.suptitle("One sample per Class ID", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
