import cv2 
import numpy as np
from glob import glob

def import_images():
    return glob('assets/*.jpg')

def show_image(window_name: str, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def process_image(image):
    color_segmented_image = segment_image_by_color(image)
    return color_segmented_image

def segment_image_by_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red = np.array([0, 70, 50])
    upper_red = np.array([25, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    masked_red = cv2.bitwise_and(image, image, mask=mask_red)
    masked_red2 = cv2.bitwise_and(image, image, mask=mask_red2)
    masked_blue = cv2.bitwise_and(image, image, mask=mask_blue)

    masked_final = cv2.bitwise_or(masked_red, masked_red2)

    return masked_final

def main():
    paths = import_images()
    img = cv2.imread(paths[10])
    img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)

    result = process_image(img_resized)

    show_image("Original", img_resized)
    show_image("Resultado", result)

if __name__ == "__main__":
    main()