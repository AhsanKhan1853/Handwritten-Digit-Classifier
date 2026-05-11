import numpy as np
from PIL import Image
import os

# Load test images and labels
images = np.load('test_images.npy')
labels = np.load('test_labels.npy')

# Create output folder
os.makedirs('sample_digits', exist_ok=True)

# Track how many of each digit we've saved
count = [0] * 10

for i in range(len(images)):
    digit = labels[i]
    
    # Only save 5 of each digit
    if count[digit] < 5:
        img = Image.fromarray(images[i].astype('uint8'))
        filename = f'sample_digits/digit_{digit}_{count[digit]}.png'
        img.save(filename)
        count[digit] += 1
    
    # Stop when we have 5 of every digit
    if all(c >= 5 for c in count):
        break

print("Done! Saved 50 images in sample_digits/")
for i in range(10):
    print(f"  Digit {i}: 5 images saved")