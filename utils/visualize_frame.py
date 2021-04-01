import cv2
import numpy as np
import matplotlib.pyplot as plt


vidcap1 = cv2.VideoCapture("inputs/datlcam1_clip1.mp4")
width1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidcap2 = cv2.VideoCapture("inputs/datlcam2_clip1.mp4")
width2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidcap2.set(cv2.CAP_PROP_POS_FRAMES, 504)
frame_count = 0

coords = [
    [(586, 369), (89, 250)],
    [(874, 377), (131, 366)],
    [(741, 333), (281, 312)],
    [(385, 289), (313, 196)],
    [(305, 255), (405, 173)],
    [(526, 265), (489, 240)],
    [(443, 236), (565, 212)],
    [(192, 209), (535, 143)],
    [(314, 194), (674, 176)],
    [(261, 177), (716, 160)],
    [(78, 162), (668, 113)],
    [(48, 148), (702, 104)],
    [(180, 150), (780, 139)],
    [(143, 138), (807, 128)],
    [(90, 121), (847, 115)],
    [(65, 113), (866, 108)],
    [(137, 208), (504, 129)],
    [(25, 203), (466, 103)],
    [(306, 364), (44, 176)]
]

while vidcap1.isOpened() and vidcap2.isOpened():

    status1, frame1 = vidcap1.read()
    status2, frame2 = vidcap2.read()

    if not status1:
        print("status1 false")
        break

    if not status2:
        print("status2 false")
        break

    frame_count += 1

    frame1 = cv2.resize(frame1, dsize=(width1//2, height1//2))
    frame2 = cv2.resize(frame2, dsize=(width2//2, height2//2))

    llcoords = np.array(
                [(200, 520), (850, 520), (105, 180), (10, 228), (10, 330)], dtype=np.int32
            ).reshape((-1, 1, 2))
    rlcoords = np.array(
                [(850, 520), (950, 316), (223, 110), (105, 180)], dtype=np.int32
            ).reshape((-1, 1, 2))

    cv2.polylines(frame1, [llcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame1, [rlcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.circle(frame1, (878, 533), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.circle(frame1, (924, 467), radius=4, color=(0, 0, 255), thickness=-1)

    llcoords = np.array(
                [(20, 185), (20, 412), (743, 146), (505, 91)], dtype=np.int32
            ).reshape((-1, 1, 2))
    rlcoords = np.array(
                [(20, 412), (20, 525), (462, 525), (934, 187), (743, 146)], dtype=np.int32
            ).reshape((-1, 1, 2))

    cv2.polylines(frame2, [llcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame2, [rlcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.circle(frame2, (5, 300), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.circle(frame2, (5, 535), radius=4, color=(0, 0, 255), thickness=-1)

    for c in coords:
        cv2.circle(frame1, (c[0][0], int(c[0][1] / 432 * 540)), radius=1, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame2, (c[1][0], int(c[1][1] / 432 * 540)), radius=1, color=(0, 0, 255), thickness=-1)

    plt.imshow(cv2.cvtColor(frame2, code=cv2.COLOR_BGR2RGB))
    plt.show()
    break

    # final_frame = np.vstack((frame1, frame2))
    # cv2.imshow("video", final_frame)

    # if frame_count == 550:
    #     cv2.waitKey(-1)

    # key = cv2.waitKey(1)

    # if key == ord('q'):
    #    break
    # elif key == ord('p'):
    #     cv2.waitKey(-1)

vidcap1.release()
vidcap2.release()
cv2.destroyAllWindows()