import cv2
import math
import matplotlib.pyplot as plt


# input_path = "inputs/kurnul_clip2.avi"
input_path = "inputs/hyderabad_clip2.avi"

vidcap = cv2.VideoCapture(input_path)

while vidcap.isOpened():
    status, frame = vidcap.read()

    frame = cv2.resize(frame, dsize=(960, 540), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pt1, pt2 = (900, 372), (200, 193)
    # pt1, pt2 = (866,384), (196, 186)
    dy, dx = (pt2[1] - pt1[1]), (pt2[0] - pt1[0])
    angle1 = math.atan2(dy, dx)
    print(angle1) # -166.68 (deg), -2.91 (rad)
    cv2.arrowedLine(frame, pt1, pt2, color=(0,0,255), thickness=3)

    pt1, pt2 = (756, 477), (93, 225)
    # pt1, pt2 = (717, 483), (100, 210)
    dy, dx = (pt2[1] - pt1[1]), (pt2[0] - pt1[0])
    angle1 = math.atan2(dy, dx)
    print(angle1) # -160.71 (deg), -2.80 (rad)
    cv2.arrowedLine(frame, pt1, pt2, color=(255,0,0), thickness=3)

    plt.imshow(frame)
    plt.show()
    break

vidcap.release()
cv2.destroyAllWindows()
