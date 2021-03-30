import cv2
import numpy as np


input_path = "inputs/hyderabad_clip8.avi"
vidcap = cv2.VideoCapture(input_path)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while vidcap.isOpened():
    status, frame = vidcap.read()

    if not status:
        break

    frame = cv2.resize(frame, dsize=(960, 540), interpolation=cv2.INTER_LINEAR)

    frame2 = cv2.cvtColor(frame, code=cv2.COLOR_BGR2YCrCb)

    # frame2_Y_eh = cv2.equalizeHist(frame2[:,:,0])
    frame2_Y_eh = clahe.apply(frame2[:, :, 0])

    frame2_eh = np.dstack((frame2_Y_eh, frame2[:, :, 1], frame2[:, :, 2]))

    frame2 = cv2.cvtColor(frame2_eh, code=cv2.COLOR_YCrCb2BGR)

    out_frame = np.hstack((frame, frame2))
    cv2.imshow("out", out_frame)

    key = cv2.waitKey(50)

    if key == ord("p"):
        cv2.waitKey(-1)
    elif key == ord("q"):
        break

vidcap.release()
cv2.destroyAllWindows()
