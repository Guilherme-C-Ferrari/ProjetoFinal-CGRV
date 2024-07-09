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
    blurred_image = blur_image(image)
    upper_half, lower_half = split_image(blurred_image)
    segmented_upper_half, segmented_lower_half = segment_image_by_color(upper_half, lower_half)
    reconnected_image = reconnect_image(segmented_upper_half, segmented_lower_half)
    return sharpen_image(reconnected_image)

def split_image(image):
    h = image.shape[0]
    half = h // 2

    upper_half = image[:half, :]
    lower_half = image[half:, :] 

    return upper_half, lower_half

def reconnect_image(upper_half, lower_half):
    reconnected_image = cv2.vconcat([upper_half, lower_half])
    return reconnected_image

def segment_image_by_color(upper_half, lower_half):
    hsv_upper = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)
    hsv_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)

    limit_lo_red1 = np.array([0, 5, 50])
    limit_up_red1 = np.array([15, 255, 255])
    limit_lo_red2 = np.array([170, 5, 50])
    limit_up_red2 = np.array([180, 255, 255])

    limit_lo_red3 = np.array([0, 80, 50])
    limit_up_red3 = np.array([15, 255, 255])
    limit_lo_red4 = np.array([170, 80, 50])
    limit_up_red4 = np.array([180, 255, 255])

    upper_half_mask_red = cv2.inRange(hsv_upper, limit_lo_red1, limit_up_red1)
    upper_half_mask_red2 = cv2.inRange(hsv_upper, limit_lo_red2, limit_up_red2)
    upper_half_masked_red = cv2.bitwise_and(upper_half, upper_half, mask=upper_half_mask_red)
    upper_half_masked_red2 = cv2.bitwise_and(upper_half, upper_half, mask=upper_half_mask_red2)
    upper_half_masked_final = cv2.bitwise_or(upper_half_masked_red, upper_half_masked_red2)

    lower_half_mask_red = cv2.inRange(hsv_lower, limit_lo_red3, limit_up_red3)
    lower_half_mask_red2 = cv2.inRange(hsv_lower, limit_lo_red4, limit_up_red4)
    lower_half_masked_red = cv2.bitwise_and(lower_half, lower_half, mask=lower_half_mask_red)
    lower_half_masked_red2 = cv2.bitwise_and(lower_half, lower_half, mask=lower_half_mask_red2)
    lower_half_masked_final = cv2.bitwise_or(lower_half_masked_red, lower_half_masked_red2)

    return upper_half_masked_final, lower_half_masked_final

def blur_image(image):
    return cv2.blur(image, (5,5))

def sharpen_image(image):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, sharpen_kernel)

def change_color(image):
    pass

def main():
    paths = import_images()
    img = cv2.imread(paths[10])
    img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)

    result = process_image(img_resized)

    show_image("Original", img_resized)
    show_image("Resultado", result)

if __name__ == "__main__":
    main()