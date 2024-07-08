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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    maskblue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])

    maskred = cv2.inRange(hsv, lower_red, upper_red)

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    maskred2 = cv2.inRange(hsv, lower_red2, upper_red2)

    lower_orange = np.array([1, 190, 200])
    upper_orange = np.array([18, 255, 255])

    maskorange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    return cv2.bitwise_and(image, image, mask=maskorange)

def main():
    paths = import_images()
    img = cv2.imread(paths[9])
    img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)

    result = process_image(img_resized)

    show_image("Original", img_resized)
    show_image("Resultado", result)


if __name__ == "__main__":
    main()