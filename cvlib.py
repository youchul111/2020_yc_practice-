import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

image_path = 'yolov3test.jpeg'
im = cv2.imread(image_path)

bbox, label, conf = cv.detect_common_objects(im)

print(bbox, label, conf)

im = draw_bbox(im, bbox, label, conf)


cv2.imwrite('result_cvlib_mine.png', im)

#cv2.imshow('result.jpg',im)
