import cv2
import numpy as np


vidcap1 = cv2.VideoCapture("inputs/datl_clip1.mp4")
width1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidcap2 = cv2.VideoCapture("inputs/datl_clip2.mp4")
width2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidcap2.set(cv2.CAP_PROP_POS_FRAMES, 500)

while vidcap1.isOpened() and vidcap2.isOpened():

    status1, frame1 = vidcap1.read()
    status2, frame2 = vidcap2.read()

    if not status1:
        print("status1 false")
        break

    if not status2:
        print("status2 false")
        break

    frame1 = cv2.resize(frame1, dsize=(width1//2, int(height1/2.5)))
    frame2 = cv2.resize(frame2, dsize=(width2//2, int(height2/2.5)))

    final_frame = np.vstack((frame1, frame2))
    cv2.imshow("video", final_frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
       break
    elif key == ord('p'):
        cv2.waitKey(-1)

vidcap1.release()
vidcap2.release()
cv2.destroyAllWindows()