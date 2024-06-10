import cv2 as cv

# 001 013 032 068
src1 = cv.imread("res2.png")
src2 = cv.imread("074.png")

c = cv.addWeighted(src1, 0.5, src2, 0.5, 1)
cv.imshow('c', c)
cv.imwrite('res3.png', c)
cv.waitKey(0)


