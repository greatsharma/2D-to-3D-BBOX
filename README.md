# 2D to 3D BBOX using Geometry

Things to change before running :-

1. video writer fps (at two places in code)
2. waitkey if running on recorded video

<br>

### Goal of POC
The goal of this POC is to map the same vechile across different cameras (Primary and Secondary camera) with high overlapping view. In the below two cameras the view is almost 100% overlapped. This mapping is useful to generate a single global track-id for the same vehicle across different camera. This result in better tracking of vehicle in understanding traffic flow. Predefined lane markers (the white stripes on road) are used to create this mapping.

In below videos left camera is Primary Camera and right camera is Secondary.

<br>

### Linear Mapping
In the below video we have two cameras for the same location but with different views. The goal here is to map the bottom-right coordinate of bbox in the primary(left) camera to secondary(right) camera. Here the RED point is the mapped coordinate, we can see that this mapping is very close.


https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/02aafd4d-a456-4a5b-9729-60b8109bea3f

<br>
In the above video we can see that the RED point in right camera is little far from the vehicle, this is bcz the view of left camera is such that the bottom-right corner of bbox will be little far from the vehicle's actual bottom. This effect is negligible for small vehicles like bikes and auto but significant for large vehicles like trucks.
<br><br>

https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/30fa2ad1-4e90-4e3d-bdd3-19490c239720

<br>

To improve the mapping quality we will use the bottom of 3d-bbox to map from left to right camera. But for this we will first convert the 2d bbox to 3d bbox.

<br>

### 2D to 3D BBOX conversion


In the below video the blue 2d bbox is the actual one inferenced from yolo detector whereas the green 3d bbox is the one constructing from the 2d bbox using simple geometry. Also the RED point here is NOT the bottom-right corner of 2d bbox but is the midpoint of bottom edge of 3d bbox.

https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/111722fe-8543-42c0-8dd5-97692f7304ef

<br>
Now in the below video we can see that the mapping of bottom edge midpoint of 3d bbox is very close. In the righ camera, the BLUE point is the actual bottom edge midpoint for that vehicle using 3d-bbox but the RED point is the mapped point from left camera. We can see that these point are close even fot large vehicles like trucks.
<br><br>

https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/d8b9f10d-34ac-4f3d-8960-6134fb49ec9d

