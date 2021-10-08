import numpy as np

def onehot(data, n):
    #n: 当前像素点为背景, 还是物体
    buf = np.zeros(data.shape + (n, ))#shape = (w,h,2)
    nmsk = np.arange(data.size)*n + data.ravel()#data.ravel()-->data.flatten()
    buf.ravel()[nmsk-1] = 1#若当前点, data为1, 则第一张图为0, 第二张图为1. 若data=0, 则第一张图为1, 第二张图为0
    return buf

if __name__ == '__main__':
    import cv2
    img_name='0.jpg'
    imgB = cv2.imread('bag_data_msk/' + img_name, 0)  # 灰度图
    imgB = cv2.resize(imgB, (160, 160))
    imgB = imgB / 255#变为0,1, 变为float64
    imgB = imgB.astype('uint8')#变为uint8
    imgB = onehot(imgB, 2)
    imgB = imgB.transpose(2, 0, 1)