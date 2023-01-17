import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

# делаем снимок с камеры
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "img_camera.jpeg"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

# Подготовка изображения
img_loaded = cv2.imread("img_camera.jpeg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Тестовое изображение")
plt.show()

# Задание 1. Перевести в grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title("Задание 1. Перевести в grayscale")
plt.show()

# Задание 2. Перевести изорбражение в hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.title("Задание 2. Перевести изорбражение в hsv")
plt.show()

# Задание 3. Повернуть изображение на 45 градусов
angle = 45 * np.pi / 180
rotate = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0]]).astype(np.float32)
rotated_img = cv2.warpAffine(img, rotate, dsize=(img.shape[1], img.shape[0]))
plt.imshow(rotated_img)
plt.title("Задание 3. Повернуть изображение на 45 градусов")
plt.show()


# Задание 4. Повернуть изображение на 30 градусов вокруг заданной точки
def rotate_around_point(px, py, angle):
    angle_rad = angle * np.pi / 180
    dx = px - (np.cos(angle_rad) * px - np.sin(angle_rad) * py)
    dy = py - (np.sin(angle_rad) * px + np.cos(angle_rad) * py)
    rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad), dx], [np.sin(angle_rad), np.cos(angle_rad), dy]]).astype(
        np.float32)

    rotate_img = cv2.warpAffine(img, rot, dsize=(img.shape[1], img.shape[0]))

    return rotate_img


rotaded = rotate_around_point(img.shape[0] / 2, img.shape[1] / 2, 30)
plt.imshow(rotaded)
plt.title("Задание 4. Повернуть изображение на 30 градусов вокруг заданной точки")
plt.show()

# Задание 5. Сместить изображение но 10 пикселей вправо
shift = np.array([[1, 0, 10], [0, 1, 0]]).astype(np.float32)
shift_img = cv2.warpAffine(img, shift, dsize=(img.shape[1], img.shape[0]))
plt.imshow(shift_img)
plt.title("Задание 5. Сместить изображение но 10 пикселей вправо")
plt.show()

# Задание 6. Изменить яркость изображения
alpha = 1
beta = 100
new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
plt.imshow(new_image)
plt.title("Задание 6. Изменить яркость изображения")
plt.show()

# Задание 7. Изменить контрасть изображения
# должен быть всегда больше 0, если меньше 1 - контраст уменьшается, если больше - увеличивается
alpha = 0.4
beta = 0
new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
plt.imshow(new_image)
plt.title("Задание 7. Изменить контрасть изображения")
plt.show()


# Задание 8 изменить баланс белого, сделать более "теплую" картинку
class WarmingFilter:

    def __init__(self):
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30, 80, 120, 192])

    def render(self, img_rgb):
        # warming filter: increase red, decrease blue
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))


warm_filter = WarmingFilter()
res = warm_filter.render(img)
plt.imshow(res)
plt.title("Задание 8 изменить баланс белого, сделать более теплую картинку")
plt.show()


# Задание 9 - изменить баланс белого, сделать более "холодную" картинку
class CoolingFilter:

    def __init__(self):
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30, 80, 120, 192])

    def render(self, img_rgb):
        # cooling filter: increase blue, decrease red
        c_r, c_g, c_b = cv2.split(img_rgb)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rgb = cv2.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))


cold_filter = CoolingFilter()
res = cold_filter.render(img)

plt.imshow(res)
plt.title("Задание 9 - изменить баланс белого, сделать более холодную картинку")
plt.show()

# Задание 10 сделать размытие изображения
kernel = np.ones((9, 9), np.float32) / 81
res = cv2.filter2D(img, -1, kernel)
plt.imshow(res)
plt.title("Задание 10 сделать размытие изображения")
plt.show()

