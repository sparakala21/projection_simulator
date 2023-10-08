# projection_simulator
Takes a 3d point cloud, a light source, intrinsic camera parameters, and a camera position and rotation and project the image from that camera. 
The full projection formula allows you to take a point in 3d space as well
as a camera, and convert it into a row and column value in the 2 dimensional image frame
the radiometric theories we use allow us to understand how the rotation and translation
of light sources affects the produced image. with the power of these 
projection and radiometric formulas we can figure out the placement and intensity
of each pixel such that it produces a coherent image.
