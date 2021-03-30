import cv2

import darknet
from detectors import BaseDetector


class VanillaYoloDetector(BaseDetector):
    def detect(self, curr_frame) -> list:
        curr_frame = cv2.cvtColor(curr_frame, code=cv2.COLOR_BGR2RGB)

        curr_frame = cv2.resize(
            curr_frame,
            (
                darknet.network_width(self.net_main),
                darknet.network_height(self.net_main),
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        darknet.copy_image_from_bytes(self.darknet_image, curr_frame.tobytes())

        yolo_detections = darknet.detect_image(
            self.net_main,
            self.meta_main,
            self.darknet_image,
            thresh=self.detection_thresh,
        )

        return self._postpreprocessing(yolo_detections)
