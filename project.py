import math
import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import csv



def illuminate(N, beta, rho, d, f, alpha, L):
    I = (((beta*rho*math.pi)/4*(d/f)**2)*np.cos(alpha)**4)*np.dot(L, N)
    return max(I,0)
def average(image):
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            if(image[i][j]<=(image[i-1, j-1]+image[i-1, j]+image[i-1, j+1]+image[i,j-1]+image[i, j+1]+image[i+1, j-1]+image[i+1, j]+image[i+1, j+1])/8):
                image[i][j]= (image[i-1, j-1]+image[i-1, j]+image[i-1, j+1]+image[i,j-1]+image[i, j+1]+image[i+1, j-1]+image[i+1, j]+image[i+1, j+1])/8
    return image

def image_projection(point, M, R, T):
    x= point[0]
    y= point[1]
    z = point[2]
    
    W = np.concatenate((R, T), axis=1 )
    # Create the input vector as a column vector
    input_vector = np.array([[x], [y], [z], [1]])
    # Perform matrix multiplications
    result = (M @ W) @ input_vector
    # # Extract the transformed (x, y, z) coordinates from the result
    transformed_x = result[0][0]
    transformed_y = result[1][0]
    transformed_z = result[2][0]
    return (int(np.round(transformed_x/transformed_z)), int(np.round(transformed_y/transformed_z)), transformed_z)


# Specify the path to your .mat file
mat_file_path = 'CV1_data.mat'

# Load the .mat file
mat_data = scipy.io.loadmat(mat_file_path)

# Now, you can access the variables stored in the .mat file as dictionary keys
# For example, if you have a variable named 'my_data' in the .mat file, you can access it like this:
Xs = mat_data['X']
Ys = mat_data['Y']
Zs = mat_data['Z']
coordinates = np.column_stack((Xs, Ys, Zs))
Nx = mat_data['Nx']
Ny = mat_data['Ny']
Nz = mat_data['Nz']
normals = np.column_stack((Nx,Ny,Nz))

R_1 = np.eye(3)
R_2 = np.array([[0.9848, 0., 0.1736],
       [0., 1., 0.],
       [-0.1736, 0., 0.9848]])

T = np.array([[-14], [-71], [1000]])

F_1 = 40
F_2 = 30
Sx = 8
Sy = 8
C_0 = 80
R_0 = 80
alpha = 30
beta = 1
rho = 1
d = 33 

L = []

L.append(np.array([0,0,-1]))
L.append(np.array([0.5574, -0.5774, -0.5774]))



M_1 = np.array([[Sx * F_1, 0, C_0],
                  [0, Sy * F_1, R_0],
                  [0, 0, 1]])
M_2 = np.array([[Sx * F_2, 0, C_0],
                  [0, Sy * F_2, R_0],
                  [0, 0, 1]])

side_length = 120

POV = []

for i in range(16):

    POV.append(np.zeros([side_length, side_length]))




cr = list()
light = list()
for i in range(len(coordinates)):
    a = image_projection(coordinates[i], M_1, R_1, T)
    POV[0][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[0])

    a = image_projection(coordinates[i], M_1, R_2, T)
    POV[1][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[0])

    a = image_projection(coordinates[i], M_2, R_1, T)
    POV[2][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[0])

    a = image_projection(coordinates[i], M_2, R_2, T)
    POV[3][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[0])

    a = image_projection(coordinates[i], M_1, R_1, T)
    POV[4][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[0])

    a = image_projection(coordinates[i], M_1, R_2, T)
    POV[5][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[0])

    a = image_projection(coordinates[i], M_2, R_1, T)
    POV[6][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[0])

    a = image_projection(coordinates[i], M_2, R_2, T)
    POV[7][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[0])


    a = image_projection(coordinates[i], M_1, R_1, T)
    POV[8][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[1])

    a = image_projection(coordinates[i], M_1, R_2, T)
    POV[9][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[1])

    a = image_projection(coordinates[i], M_2, R_1, T)
    POV[10][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[1])

    a = image_projection(coordinates[i], M_2, R_2, T)
    POV[11][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_1, alpha, L[1])

    a = image_projection(coordinates[i], M_1, R_1, T)
    POV[12][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[1])

    a = image_projection(coordinates[i], M_1, R_2, T)
    POV[13][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[1])

    a = image_projection(coordinates[i], M_2, R_1, T)
    POV[14][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[1])

    a = image_projection(coordinates[i], M_2, R_2, T)
    POV[15][a[1], a[0]] = illuminate(normals[i], beta, rho, d, F_2, alpha, L[1])
    


    


POV[0] = average(average(POV[0]))


for i in range(len(POV)):

    full = np.max(POV[i])

    POV[i] = np.round((POV[i]/full)*255).astype(int)

    matplotlib.image.imsave('POV{}.png'.format(i), POV[i], cmap = 'gray')








