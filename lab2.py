import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

                #Right 0  Right-Up 1   Up 2   Left-Up 3   Left 4   Left-Down 5   Down 6   Right-Down 7
directions = [ (0, 1),    (-1, 1),     (-1, 0),  (-1, -1), (0, -1),  (1, -1),  (1, 0),   (1, 1)]

print("Loading MNIST...")
(x_train, y_train), _ = mnist.load_data()

digit_to_find = int(input("Enter digit label (0-9): "))
indices = np.where(y_train == digit_to_find)[0]

if len(indices) == 0:
    print("Digit not found.")
    exit()

image = x_train[indices[0]]

binary = (image > 128).astype(np.uint8) * 255

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) == 0:
    print("No contour found!")
    exit()

contour = max(contours, key=cv2.contourArea)

boundary_pixels = [(pt[0][1], pt[0][0]) for pt in contour]

def get_chain_code(pixels):
    chain_code = []
    for i in range(len(pixels)-1):
        r1, c1 = pixels[i]
        r2, c2 = pixels[i+1]
        dr = r2 - r1
        dc = c2 - c1
        for idx, (dx, dy) in enumerate(directions):
            if (dr, dc) == (dx, dy):
                chain_code.append(idx)
                break
    return chain_code

result = get_chain_code(boundary_pixels)


print("\nFinal Chain Code:")
print(result)

plt.imshow(image, cmap='gray')
plt.title(f"MNIST Digit: {digit_to_find}")
plt.axis('off')
plt.show()