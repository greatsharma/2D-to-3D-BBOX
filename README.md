# 2D to 3D BBOX using Geometry

Things to change before running :-

1. video writer fps (at two places in code)
2. waitkey if running on recorded video

<br>

### Linear Mapping
In the below video we have two cameras for the same location but with different views. The goal here is to map the bottom-right coordinate of bbox in the primary(left) camera to secondary(right) camera. This mapping will help us to give a common global ID to the same vehicle in two different cameras. Here the RED point is the mapped coordinate, we can see that this mapping is very close. Also predefined lane markers (the white stripes on road) are used to create this mapping.

https://github.com/greatsharma/2D-to-3D-BBOX/assets/32649388/30fa2ad1-4e90-4e3d-bdd3-19490c239720

