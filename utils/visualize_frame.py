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
                [(200, 520), (652, 495), (72, 197), (10, 228), (10, 330)], dtype=np.int32
            ).reshape((-1, 1, 2))
    mlcoords = np.array(
                [(652, 495), (803, 444), (129, 165), (72, 197)], dtype=np.int32
            ).reshape((-1, 1, 2))
    rlcoords = np.array(
                [(803, 444), (916, 309), (223, 110), (129, 165)], dtype=np.int32
            ).reshape((-1, 1, 2))

    cv2.polylines(frame1, [llcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame1, [mlcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame1, [rlcoords], isClosed=True, color=(0, 0, 0), thickness=2)

    llcoords = np.array(
                [(4, 188), (89, 313), (690, 134), (505, 91)], dtype=np.int32
            ).reshape((-1, 1, 2))
    mlcoords = np.array(
                [(89, 313), (282, 392), (806, 158), (690, 134)], dtype=np.int32
            ).reshape((-1, 1, 2))
    rlcoords = np.array(
                [(282, 392), (581, 444), (934, 187), (806, 158)], dtype=np.int32
            ).reshape((-1, 1, 2))

    cv2.polylines(frame2, [llcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame2, [mlcoords], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame2, [rlcoords], isClosed=True, color=(0, 0, 0), thickness=2)

    plt.imshow(cv2.cvtColor(np.vstack((frame1, frame2)), code=cv2.COLOR_BGR2RGB))
    plt.show()
    break

vidcap1.release()
vidcap2.release()
cv2.destroyAllWindows()