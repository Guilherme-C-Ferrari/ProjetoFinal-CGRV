import cv2 
import numpy as np
from glob import glob

# Função que retorna todos os caminhos das imagens em uma lista
def import_images():
    return glob('assets/*.jpg')

# Função que mostras as imagens em janelas
def show_image(window_name: str, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

# Função que realiza o fluxo de PDI a partir das outraas funções existente, retornando a imagem final
def process_image(image):
    upper_half, lower_half = split_image(image)
    segmented_upper_half, segmented_lower_half = segment_image_by_color(upper_half, lower_half)
    reconnected_image = reconnect_image(segmented_upper_half, segmented_lower_half)
    changed_color_image = change_color(reconnected_image)
    return apply_color_on_original(image, changed_color_image)

# Função que divide a imagem, gerando uma parte de cima e uma de baixo
def split_image(image):
    h = image.shape[0]
    half = h // 2

    upper_half = image[:half, :]
    lower_half = image[half:, :] 

    return upper_half, lower_half

# Função que reconecta a imagem dividida
def reconnect_image(upper_half, lower_half):
    reconnected_image = cv2.vconcat([upper_half, lower_half])
    return reconnected_image

# Função que segmenta a imagem em cima das cores escolhidas, retornando as duas partes segmentadas
def segment_image_by_color(upper_half, lower_half):
    # É gerada uma versão das duas partes em HSV
    hsv_upper = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)
    hsv_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)

    # São estabelecidos os limites mínimos e máximos de cor, sendoe estabelecidos quatro grupos de limites, dois para a parte de cima e dois para a parte de baixo
    limit_lo_red1 = np.array([0, 10, 50])
    limit_up_red1 = np.array([15, 255, 255])
    limit_lo_red2 = np.array([150, 10, 50])
    limit_up_red2 = np.array([180, 255, 255])

    limit_lo_red3 = np.array([0, 75, 50])
    limit_up_red3 = np.array([15, 255, 255])
    limit_lo_red4 = np.array([170, 75, 50])
    limit_up_red4 = np.array([180, 255, 255])

    # São criadas duas máscaras para cada parte, que geram duas imagens segmentadas que então são unidas e retornadas
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

# Recebe a imagem reunida, gerando uma versão HSV, alterando o valor de hue proporcionalmente nessa imagem, e devolvendo ela a um formato BGR
def change_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    h = (h + 70) % 180
    hsv_modificado = cv2.merge([h,s,v])
    bgr = cv2.cvtColor(hsv_modificado, cv2.COLOR_HSV2BGR)
    
    return bgr

# Aplica a cor na imagem original a partir da união da imagem alterada com uma versão segmentada de forma invertida a imagem original
def apply_color_on_original(original_image, changed_color_image):
    null_value = np.array([0,0,0])
    inverted_mask = cv2.inRange(changed_color_image, null_value, null_value)
    inverted_masked = cv2.bitwise_and(original_image, original_image, mask=inverted_mask)

    return cv2.bitwise_or(inverted_masked, changed_color_image)

# Chama a função de importação de imagens e inicia um loop para alterar todas as imagens listadas
def main():
    paths = import_images()

    for path in paths:
        img = cv2.imread(path)

        # Diminui o tamanho da imagem original proporcionalmente
        img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)

        result = process_image(img_resized)

        show_image("Original", img_resized)
        show_image("Resultado", result)

if __name__ == "__main__":
    main()