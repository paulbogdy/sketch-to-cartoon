from PIL import Image
import numpy as np
from skimage import morphology
from skimage.util import img_as_bool

# Load the image
img = Image.open('sketch_test.png')

# Resize to 512x512
img = img.resize((512, 512))

# Convert to grayscale
img = img.convert('L')

# Binarize
threshold = 128
img = img.point(lambda p: p > threshold and 255)

# Invert colors
img = Image.frombytes('L', img.size, bytes(255 - b for b in img.tobytes()))

# Convert the image to a NumPy array for skeletonization
img_array = np.array(img)

# Perform skeletonization
skeleton = morphology.skeletonize(img_array == 0) # skeletonize the black parts

# Convert the skeleton back to a binary PIL image and invert back
img = Image.fromarray(np.where(skeleton, 0, 255).astype(np.uint8))

# Convert image to RGBA and make white color (now our lines) transparent
img = img.convert('RGBA')
data = img.getdata()

new_data = []
for item in data:
    # change all white (also shades of whites) pixels to black
    if item[0] > 200 and item[1] > 200 and item[2] > 200:
        new_data.append((0, 0, 0, 0))
    else:
        new_data.append((0, 0, 0, 255))

img.putdata(new_data)

# Save the image
img.save('sketch_test_transparent.png', 'PNG')
