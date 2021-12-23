import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from detectors import BaseDetector


TRT_LOGGER = trt.Logger(trt.Logger.Severity.ERROR)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtYoloDetector(BaseDetector):
    def _allocate_buffers(self, engine, batch_size):
        # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dims = engine.get_binding_shape(binding)

            # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1

            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def _do_inference(self, bindings, inputs, outputs, stream):
        """
        This function is generalized for multiple inputs/outputs.
        inputs and outputs are expected to be lists of HostDeviceMem objects.
        """

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

        # Run inference.
        self.context.execute_async(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

        # Synchronize the stream
        stream.synchronize()

        # Return only the host outputs.
        return [out.host for out in outputs]

    def _nms_cpu(self, boxes, confs, nms_thresh, min_mode=False):
        # print(boxes.shape)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)

    def detect(self, curr_frame) -> list:
        curr_frame = cv2.cvtColor(curr_frame, code=cv2.COLOR_BGR2RGB)

        curr_frame = cv2.resize(
            curr_frame,
            (
                self.yolo_width,
                self.yolo_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        curr_frame = np.transpose(curr_frame, (2, 0, 1)).astype(np.float32)
        curr_frame = np.expand_dims(curr_frame, axis=0)
        curr_frame /= 255.0
        curr_frame = np.ascontiguousarray(curr_frame)

        inputs, outputs, bindings, stream = self.buffers
        inputs[0].host = curr_frame

        trt_outputs = self._do_inference(
            bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, self.num_classes)

        # [batch, num, 1, 4]
        box_array = trt_outputs[0]

        # [batch, num, num_classes]
        confs = trt_outputs[1]

        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        argwhere = max_conf[0] > self.detection_thresh
        l_box_array = box_array[0, argwhere, :]
        l_max_conf = max_conf[0, argwhere]
        l_max_id = max_id[0, argwhere]

        trt_detections = []
        # nms for each class
        for j in range(self.num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = self._nms_cpu(ll_box_array, ll_max_conf, nms_thresh=0.5)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    trt_detections.append(
                        (
                            self.class_names[ll_max_id[k]],
                            ll_max_conf[k],
                            (
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                            ),
                        )
                    )

        return self._postpreprocessing(trt_detections)
