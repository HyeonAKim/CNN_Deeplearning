#-*-coding:utf-8 -*-
import cv2

img = cv2.imread('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Product\\Preprocess\\test\\test_test3.jpeg')
lower_reso = cv2.pyrDown(img) # 원본 이미지의 1/4 사이즈 # 2000
# lower_reso2 = cv2.pyrDown(lower_reso) # 1000
# lower_reso4 = cv2.pyrDown(lower_reso2) # 500

# higher_reso = cv2.pyrUp(img) #원본 이미지의 4배 사이즈

# cv2.imshow('img', img)
# cv2.imshow('lower', lower_reso)
# cv2.imshow('lowe_250', lower_reso2)
# cv2.imshow('lower', lower_reso)
# print(lower_reso4.shape[:2])

# 이미지 color 변경
imgray = cv2.cvtColor(lower_reso,cv2.COLOR_BGR2GRAY)
# imgray = cv2.Canny(imgray, 100, 200,3)
#
# ret,thresh = cv2.threshold(imgray,200,255,cv2.THRESH_BINARY_INV)
#
# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(imgray, contours, 1, (0,255,0))


lower_reso = cv2.pyrDown(imgray) # 원본 이미지의 1/4 사이즈 # 2000
lower_reso2 = cv2.pyrDown(lower_reso) # 원본 이미지의 1/4 사이즈 # 2000
lower_reso4 = cv2.pyrDown(lower_reso2) # 원본 이미지의 1/4 사이즈 # 2000

# contrast
# lower_reso4[:,:,2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in lower_reso4[:,:,2]]
# cv2.imshow('contrast', cv2.cvtColor(lower_reso4, cv2.COLOR_HSV2BGR))


cv2.imwrite('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Product\\Preprocess\\test\\result3.png', lower_reso2)
cv2.imwrite('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Product\\Preprocess\\test\\test_cv3.jpeg',lower_reso4)
# cv2.imshow('higher', higher_reso)

cv2.waitKey(0)

cv2.destroyAllWindows()