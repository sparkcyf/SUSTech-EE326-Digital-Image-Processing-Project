import numpy as np
import cv2
import matplotlib.pyplot as plt

input_img = cv2.imread('1492627187263118217/1.jpg')

def transform(input_img,num):
    rows,cols,ch = input_img.shape
    print(input_img.shape)

    # pts1 = np.float32([[629,250],[0,445],[705,250],[1279,430]])
    pts1 = np.float32([[450,300],[-851,720],[870,300],[2057,720]])
    pts2 = np.float32([[0,0],[0,300],[300,0],[300,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(input_img,M,(300,300))
    dst_extract = abs_thresh(dst, sobel_kernel=3, mag_thresh=(100,200), direction='x')

    plt.subplot(131),plt.imshow(input_img),plt.title('Input')
    plt.subplot(132), plt.imshow(dst), plt.title('Output')
    plt.subplot(133),plt.imshow(dst_extract),plt.title('op_extracted')
    # plt.show()
    plt.savefig("op/op_reg"+ str(num) +".png", dpi=300)


def abs_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), return_grad=False, direction='x'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = None
    scaled_sobel = None

    # Sobel x
    if direction.lower() == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    # Sobel y
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in y

    if return_grad == True:
        return grad

    abs_sobel = np.absolute(grad)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1

    return grad_binary

for i in range(1,20):
    input_img = cv2.imread('1492627187263118217/' + str(i) + '.jpg')
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    transform(input_img,i)