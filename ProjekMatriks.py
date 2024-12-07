import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    plt.figure(figsize=(6,6))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

image = cv2.imread('D:/tess/PY/orangrandom.jpeg')  
display_image('Gambar Asli', image)

# Smooth
kernel_smooth = np.ones((5,5), np.float32)/25
smooth = cv2.filter2D(image, -1, kernel_smooth)
display_image('Smooth', smooth)

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (15,15), 5)
display_image('Gaussian Blur', gaussian_blur)

# Sharpen
kernel_sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
sharpen = cv2.filter2D(image, -1, kernel_sharpen)
display_image('Sharpen', sharpen)

# Mean Removal
kernel_mean_removal = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
mean_removal = cv2.filter2D(image, -1, kernel_mean_removal)
display_image('Mean Removal', mean_removal)

# Emboss
kernel_emboss = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
emboss = cv2.filter2D(image, -1, kernel_emboss)
display_image('Emboss', emboss)

# Edge Detection (Sobel)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edge_detection = cv2.magnitude(sobel_x, sobel_y)
edge_detection = cv2.convertScaleAbs(edge_detection)
display_image('Edge Detection', edge_detection)