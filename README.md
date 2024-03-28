# 2D to 3D BBOX using Geometry

Things to change before running :-

1. video writer fps (at two places in code)
2. waitkey if running on recorded video

<br>

### Linear Mapping
In the below video we have two cameras for the same location but with different views. The goal here is to map the bottom-right coordinate of bbox in the primary(left) camera to secondary(right) camera. This mapping will help us to give a common global ID to the same vehicle in two different cameras. Here the RED point is the mapped coordinate, we can see that this mapping is very close. Also predefined lane markers (the white stripes on road) are used to create this mapping.


https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/02aafd4d-a456-4a5b-9729-60b8109bea3f


https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/30fa2ad1-4e90-4e3d-bdd3-19490c239720

https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/aab9ec23-ed5f-4f45-a3eb-3de70ddaac0c

<br>

### 2D to 3D BBOX conversion

In the below video the blue 2d bbox is the actual one inferenced from yolo detector whereas the green 3d bbox is the one constructing from the 2d bbox using simple geometry. Also the RED point here is NOT the bottom-right corner of 2d bbox but is the midpoint of bottom edge of 3d bbox.

https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/111722fe-8543-42c0-8dd5-97692f7304ef


https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/d8b9f10d-34ac-4f3d-8960-6134fb49ec9d

