import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = np.concatenate((x_train, x_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

# Convert images to binary
# Pixels > 128 -> 1 else 0
binary_images = (images > 128).astype(np.uint8)

vertical_blocks = int(input("Enter number of vertical blocks: "))
horizontal_blocks = int(input("Enter number of horizontal blocks: "))

img_height = 28
img_width = 28

block_h = img_height // vertical_blocks
block_w = img_width // horizontal_blocks


def compute_centroid(block):
    ys, xs = np.where(block == 1)
    if len(xs) == 0:
        return 0.0, 0.0
    cx = np.mean(xs)
    cy = np.mean(ys)
    return cx, cy


feature_vectors = []

for img in binary_images:
    features = []
    for i in range(vertical_blocks):
        for j in range(horizontal_blocks):
            start_y = i * block_h
            end_y = start_y + block_h
            start_x = j * block_w
            end_x = start_x + block_w
            block = img[start_y:end_y, start_x:end_x]
            cx, cy = compute_centroid(block)
            features.append(cx)
            features.append(cy)
    feature_vectors.append(features)
feature_vectors = np.array(feature_vectors)

output_file = "output.txt"

with open(output_file, "w") as f:
    print("Feature vectors shape:", feature_vectors.shape)
    print("Labels shape:", labels.shape)
    f.write("Feature vectors shape: " + str(feature_vectors.shape) + "\n")
    f.write("Labels shape: " + str(labels.shape) + "\n")
    for i in range(len(feature_vectors)):
        f.write(f"\nImage {i + 1}\n")
        f.write("Label: " + str(labels[i]) + "\n")
        f.write("Feature vector: " + str(feature_vectors[i].tolist()) + "\n")
    for i in range(5):
        print(f"\nImage {i + 1}")
        print("Label:", labels[i])
        print("Feature vector:", feature_vectors[i].tolist())
print(f"\nAll data saved to {output_file}")