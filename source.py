from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


def bgr_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def contours_to_sorted_rects(contours):
    biggest_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rects = [cv2.minAreaRect(cnt) for cnt in biggest_contours]
    return rects


def contours_to_sorted_bounding_rects(contours):
    biggest_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    brects = [cv2.boundingRect(contour) for contour in biggest_contours]
    return brects


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


def get_by_indexes(arr, indexes):
    return np.array(arr)[indexes]


IMG_NUM = 7
WHITE_SENSITIVITY = 70
BAR_WIDTH = 800
DIGIT_SIZE = 28

if __name__ == '__main__':
    img = cv2.imread('./bars_img/bar_' + str(IMG_NUM) + '.jpg')

    hsv = bgr_to_hsv(img)

    lower_white = np.array([78, 0, 255 - WHITE_SENSITIVITY])
    upper_white = np.array([138, WHITE_SENSITIVITY, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    img_masked = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = contours_to_sorted_rects(contours)
    boxes = [np.int0(cv2.boxPoints(rect)) for rect in rects]

    if not len(rects):
        print('Barcode not found')
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        exit(1)

    # crop barcode
    barcode_crop = crop_rect(img_masked, rects[0])
    bar_w, bar_h = barcode_crop.shape[1], barcode_crop.shape[0]
    ratio = bar_w / bar_h
    print(f'Barcode size: ({bar_w}, {bar_h}), ratio: {round(ratio, 2)}')

    # resize barcode
    scale = BAR_WIDTH / bar_w
    new_bar_w = BAR_WIDTH
    new_bar_h = np.int0(bar_h * scale)
    barcode_crop = cv2.resize(barcode_crop, (new_bar_w, new_bar_h))

    # draw rects and put rotation angle as text
    cv2.drawContours(img, boxes, 0, (0, 0, 255), 2)
    cv2.putText(img, str(round(rects[0][2])), np.int0(rects[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)

    # cv2.imshow('img', img)
    # cv2.imshow('barcode', barcode_crop)

    bar_gray = cv2.cvtColor(barcode_crop, cv2.COLOR_BGR2GRAY)
    _, bar_thresh = cv2.threshold(bar_gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contours in barcode
    contours, _ = cv2.findContours(bar_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_CONTOUR_AREA = 70.0  # TODO: д.б. адаптивная величина
    contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_CONTOUR_AREA]

    # convert contours to bounding rects
    rects = contours_to_sorted_rects(contours)
    bounding_rects = contours_to_sorted_bounding_rects(contours)

    # filter rects by aspect ratio and square
    filter_indexes = np.argwhere(
        np.round(np.logical_and(
            np.array([max(rect[1]) / min(rect[1]) for rect in rects]) < 3.5,
            np.array([rect[1][0] * rect[1][1] for rect in rects]) <= 1000 * scale ** 2,
        ))
    ).flatten()
    rects = get_by_indexes(rects, filter_indexes)
    bounding_rects = get_by_indexes(bounding_rects, filter_indexes)

    # sort rects from lowest position to highest  
    sort_indexes = np.argsort([rect[0][1] for rect in rects])[::-1]

    # barcode is 13 digits
    DIGITS_COUNT = 13
    rects = get_by_indexes(rects, sort_indexes)[:DIGITS_COUNT]
    bounding_rects = get_by_indexes(bounding_rects, sort_indexes)[:DIGITS_COUNT]

    # sort from left to right position
    sort_indexes = np.argsort([rect[0][0] for rect in rects])
    rects = get_by_indexes(rects, sort_indexes)
    bounding_rects = get_by_indexes(bounding_rects, sort_indexes)

    # print('square', [w * h for center, (w, h), a in rects])
    # print('top offset', [rect[0][1] for rect in rects])
    # print('width', [rect[1][0] for rect in rects])
    # print('height', [rect[1][1] for rect in rects])
    # print('angle', [rect[2] for rect in rects])
    # print('ration', [np.round(max(rect[1]) / min(rect[1]), 2) for rect in rects])

    for i, rect in enumerate(rects):
        cv2.putText(barcode_crop, str(i), np.int0(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # get result image with barcode digits
    max_h = max(bounding_rects, key=lambda rect: rect[3])[3]
    digits = []
    for x, y, w, h in bounding_rects:
        cv2.rectangle(barcode_crop, (x, y), (x + w, y + h), (0, 0, 255), 1)
        roi = bar_thresh[y:y + h, x:x + w]

        k = max(w, max_h + round(0.4 * max_h))
        roi = cv2.resize(roi, (w, max_h))
        h = max_h
        padding = (max((k - h) // 2, 0), max((k - w) // 2, 0))
        roi = cv2.copyMakeBorder(
            roi,
            padding[0], padding[0], padding[1], padding[1],
            cv2.BORDER_CONSTANT,
            None,
            value=0
        )
        roi = cv2.resize(roi, (28, 28))
        roi = cv2.erode(roi, np.ones((2, 2), np.uint8), iterations=1)

        digits.append(roi)

    digits_img = np.hstack(digits)

    # plt.imshow(digits[8], cmap='Greys')
    # plt.show()

    cv2.imshow('barcode', barcode_crop)
    cv2.imshow('digits_img', digits_img)
    cv2.waitKey(0)
