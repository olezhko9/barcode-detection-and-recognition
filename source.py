import cv2
import numpy as np


def bgr_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def contours_to_boxes_filtered(contours):
    biggest_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rects = [cv2.minAreaRect(cnt) for cnt in biggest_contours]
    # rects = list(filter(lambda rect: (rect[2] < 10.0 or rect[2] > 80.0) and rect[1][0] < rect[1][1], rects))
    boxes = [np.int0(cv2.boxPoints(rect)) for rect in rects]
    return boxes, rects


def crop_rect(img, rect):
    center, size, angle = rect
    box_w, box_h = size

    height, width, _ = img.shape

    rotate_angle = angle
    if box_w < box_h:
        rotate_angle = angle - 90.0

    M = cv2.getRotationMatrix2D(center, rotate_angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    cropped_w = int(box_h if box_h > box_w else box_w)
    cropped_h = int(box_h if box_h < box_w else box_w)
    img_crop = cv2.getRectSubPix(img_rot, (cropped_w, cropped_h), center)

    return img_crop


WHITE_SENSITIVITY = 70

if __name__ == '__main__':
    img = cv2.imread('./bars_img/bar_5.jpg')

    hsv = bgr_to_hsv(img)

    lower_white = np.array([78, 0, 255 - WHITE_SENSITIVITY])
    upper_white = np.array([138, 255 - WHITE_SENSITIVITY, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # img = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, rects = contours_to_boxes_filtered(contours)

    if not len(rects):
        print('Barcode not found')
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        exit(1)

    barcode_crop = crop_rect(img, rects[0])

    cv2.drawContours(img, boxes, 0, (0, 0, 255), 2)
    cv2.putText(img, str(round(rects[0][2])), np.int0(rects[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.imshow('barcode', barcode_crop)
    cv2.waitKey(0)
