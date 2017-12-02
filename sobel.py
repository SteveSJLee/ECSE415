import cv2
import numpy as np

class Sobel:
    def __init__(self, img):
        self.img = img
        self.sobelX = None
        self.sobelY = None
        self.sb = None

    # use kernel to filter the input image
    def kernel_filter (self, kernel, limit='no'):
        height, width = self.img.shape
        kernel_height, kernel_width = kernel.shape

        img_filter = np.zeros(((height+2), (width+2)), np.float32)
        window = np.zeros((kernel_height, kernel_width), np.float32)
        value = np.zeros((kernel_height, kernel_width), np.float32)

        for row in range(1, height-1):
            for col in range(1, width-1):
                window = self.img[(row - 1): (row + 2), (col - 1): (col + 2)]
                value = window*kernel
                            #limit pixel value
                if limit == "yes":
                    img_filter[row, col] = int(value.sum())
                else:
                    img_filter[row, col] = int(value.sum()/9)

        return img_filter

    def filter(self):
        filterX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        filterY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        filterG = np.array([(1, 2, 1), (2, 4, 2), (1, 2, 1)], np.float32)/16

        #self.img = self.kernel_filter(filterG, 'yes')
        #self.sobelX = self.kernel_filter(filterX)
        #self.sobelY = self.kernel_filter(filterY)
        self.img = cv2.blur(img, (3,3))
        self.sobelX = self.kernel_filter(filterX)
        self.sobelY = self.kernel_filter(filterY)

        sb = (self.sobelX**2 + self.sobelY**2)**0.5
        self.sb = sb

    def threshold(self):
        a = np.copy(self.sb)
        th = 10
        a[a >= th] = 0
        a[a != 0] = 255
        self.sb = a

    def sobel(self):
        self.filter()
        self.threshold()
        return self.sb


img = cv2.imread('sample.jpg', 0)

sb = Sobel(img)

img = sb.sobel()

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()