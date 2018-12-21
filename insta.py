# python insta.py -l load_path -s store_path

import cv2
import argparse
import numpy as np


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--load', required=True,
                    help='load path for loading img')
    ap.add_argument('-s', '--store', required=True,
                    help='load path for storing img')
    args = vars(ap.parse_args())

    if not args['load'] or not args['store']:
        ap.error("Please enter path")
    return args


def main():

    while True:
        args = get_arguments()
        img = str(args['load'])
        image = cv2.imread(img)

        frame_to_thresh = image.copy()
        thresh = cv2.inRange(frame_to_thresh, (1, 1, 1), (250, 250, 250))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours in the mask and initialize the current
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            crop_img = image[y+1:y + h, x:x + w]
            cv2.imwrite(str(args['store']), crop_img)
        return 0
        # else error no image detected


if __name__ == '__main__':
    main()
