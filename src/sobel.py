import cv2
import numpy as np

class SB:
    def __init__(self):
        #self.img = img
        self.sobelX = None
        self.sobelY = None
        self.sb = None

    # use kernel to filter the input image
    def kernel_filter (self, kernel):
        height, width = self.img.shape
        kernel_height, kernel_width = kernel.shape

        img_pad = np.zeros(((height+2), (width+2)), np.float32)
        img_pad[1:height+1, 1:width+1] = self.img

        window = np.zeros((kernel_height, kernel_width), np.float32)
        value = np.zeros((kernel_height, kernel_width), np.float32)
        img_filter = np.zeros((height, width))

        for row in range(1, height):
            for col in range(1, width):
                window = img_pad[(row - 1): (row + 2), (col - 1): (col + 2)]
                value = window*kernel
                img_filter[row-1, col-1] = int(value.sum())

        return img_filter

    def filter(self):
        filterX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        filterY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        filterG = np.array([(1, 2, 1), (2, 4, 2), (1, 2, 1)], np.float32)/16

        self.sobelX = self.kernel_filter(filterX)
        self.sobelY = self.kernel_filter(filterY)

        sb = cv2.magnitude(self.sobelX, self.sobelY)
        #(self.sobelX**2 + self.sobelY**2)**0.5
        self.sb = sb#cv2.GaussianBlur(sb, (5,5), 0)

    def threshold(self, th):
        img_thr = np.copy(self.sb)
        img_thr[img_thr >= th] = 255
        img_thr[img_thr != 255] = 0

        self.sb = img_thr

    def sobel(self, img, th=100):
        self.img = img
        self.filter()
        self.threshold(th)
        return self.sb




def apply_sobel(img, ksize, thres=None):
    """
    Apply Sobel operator [ksizexksize] to image.
    @param  img:    input image
    @param  ksize:  Sobel kernel size
                    @pre odd integer >= 3
    @param  thres:  binary threshold, if None do not threshold
                    @pre integer >= 0 & <= 255
    
    @return:    image of Sobel magnitude
    """
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    Im = cv2.magnitude(Ix, Iy)
    if thres is not None:
        _, It = cv2.threshold(Im, thres, 1, cv2.THRESH_BINARY)
        return It
    else:
        return Im


# example
img = cv2.imread('eyes_search.png', 0)
#img = cv2.resize(img, (150,200))
sb = SB()
img = sb.sobel(img)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
