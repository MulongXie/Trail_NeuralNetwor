import cv2


def label(img, label):
    print(label)
    for o in label:
        print(o)
        top_left = (int(o['xmin']), int(o['ymin']))
        bottom_right = (int(o['xmax']), int(o['ymax']))

        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)

