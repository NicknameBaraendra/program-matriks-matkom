import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    plt.figure(figsize=(6,6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

image = cv2.imread('C:/Users/RYZEN/OneDrive/Pictures/testing/WhatsApp Image 2024-12-07 at 17.32.47_03df35bd.jpg')
display_image('Gambar Asli', image)

# 1. Smooth
kernel_smooth = np.ones((5,5),np.float32)/25
smooth = cv2.filter2D(image, -1, kernel_smooth)
display_image('Smooth', smooth)

# 2. Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (9, 9), 5)
display_image('Gaussian Blur', gaussian_blur)

# 3. Sharpen
kernel_sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
sharpen = cv2.filter2D(image, -1, kernel_sharpen)
display_image('Sharpen', sharpen)

# 4. Mean Removal
kernel_mean_removal = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
mean_removal = cv2.filter2D(image, -1, kernel_mean_removal)
display_image('Mean Removal', mean_removal)

# 5. Emboss kiri
kernel_emboss_left = np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
emboss_left = cv2.filter2D(image, -1, kernel_emboss_left)
display_image('Emboss Kiri', emboss_left)

# 6. Emboss kanan
kernel_emboss_right = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
emboss_right = cv2.filter2D(image, -1, kernel_emboss_right)
display_image('Emboss Kanan', emboss_right)

# 7. Emboss atas
kernel_emboss_top = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
emboss_top = cv2.filter2D(image, -1, kernel_emboss_top)
display_image('Emboss Atas', emboss_top)

# 8. Emboss bawah
kernel_emboss_bottom = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
emboss_bottom = cv2.filter2D(image, -1, kernel_emboss_bottom)
display_image('Emboss Bawah', emboss_bottom)

# 9. Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edge_detection = cv2.magnitude(sobel_x, sobel_y)
edge_detection = cv2.convertScaleAbs(edge_detection)
display_image('Edge Detection', edge_detection)