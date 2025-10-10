import cv2
import pytesseract
from matplotlib import pyplot as plt

image_path = "/mnt/data/08d702e9-fb88-4a5b-98ce-9ad0bc19afd9.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Grayscale Image:")
cv2_imshow(gray)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

extracted_text = pytesseract.image_to_string(image_rgb)
print(" Extracted Text:\n")
print(extracted_text)

data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)

n_boxes = len(data['level'])
for i in range(n_boxes):
    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Image with Text Bounding Boxes")
plt.axis("off")
plt.show()

