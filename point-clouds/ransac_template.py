#!/usr/bin/env python
from platform import dist
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

import random as rd
import math
###YOUR IMPORTS HERE###

def get_random_points(num_of_pts, pc):

    rand_pts = []
    for i in range(num_of_pts):
        # TODO: random a (x,y,z) in 3D space
        index = rd.randint(0, len(pc)-1)
        rand_pts.append(np.squeeze(np.asarray(pc[index])))
    # print("rand_pts: ",rand_pts)
    return rand_pts

def model_fit(input_pts):
    # TODO: Fit a plane to these rand_pts! (3 points)
    size = len(input_pts)

    if size == 3:
        # print("size = 3")
        rand_pts = input_pts
        p1 = rand_pts[0]
        p2 = rand_pts[1]
        p3 = rand_pts[2]

    else:
        # print("random indices!")
        indices = []
        used = np.full((size), False, dtype=bool)
        for _ in range(3):
            index = rd.randint(0, size-1)
            while used[index] == True:
                index = rd.randint(0, size-1)
            indices.append(index)
            used[index] = True
        p1 = input_pts[indices[0]]
        p2 = input_pts[indices[1]]
        p3 = input_pts[indices[2]]

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    # print('The equation of this plane is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    model_params = [a, b, c, d]
    return model_params

    # Function to find distance

def distance_to_plane(point, plane_params):
    # print("type: ", type(point))
    # print("point : ",point)
    # print("plane params: ",plane_params)
    a, b, c, d = plane_params
    x1, y1, z1 = point
    ##### KEY : distance to a plane ####
    d = abs((a * x1 + b * y1 + c * z1 - d))
    e = (math.sqrt(a * a + b * b + c * c))
    # print("Perpendicular distance is", d/e)
    return d/e

def Error(pts, model_param):
    # TODO: sum of squared residuals from p to plane(model_param)
    err = 0
    for pt in pts:
        err += pow(distance_to_plane(pt, model_param), 2) # TODO can use 2
    return err


def RANSAC(pc, num_of_pts=3, N=20, ITER=2000, threshold=0.1):
    # Model type: a plane !
    err_best = 10000
    model_params_best = 0
    initialized = False
    pt_in_plane = pc[0]  # arbitrary point

    for it in range(ITER):
        if it%100 == 0:
            print("iteration: ",it)
           # Pick a random subset of points (3 pts)
        rand_pts = get_random_points(num_of_pts, pc)

           # Fit "a plane" to these points
        model_params = model_fit(rand_pts)
        if not initialized:
            model_params_best = model_params
            initialized = True

        plane = []
        for pt in rand_pts:
            plane.append(pt)  # np.array

        # Make consensus set
        C = []
        for pt in pc:
            pt = np.squeeze(np.asarray(pt))
            found = False
            for rand_pt in plane:
                if np.array_equal(rand_pt, pt):
                    found = True
            # if pt not in plane:
            if not found:
                # if error less than threshold :
                if distance_to_plane(pt, model_params) <= threshold:
                    # print("dis: ", distance_to_plane(pt, model_params)," <= threshold: ", threshold)
                    C.append(pt)

        if len(C) >= N:
            # print("Number of Consensur: ",len(C))
            Union = plane + C
            # print("Union: ", Union)
            model_params = model_fit(Union)
            err_new = Error(Union, model_params)
            if err_new < err_best:
                print("iter: ", it, " update err_best from ",
                      err_best, " to ", err_new)
                err_best = err_new
                model_params_best = model_params
                pt_in_plane = rand_pts[0]
            

    return pt_in_plane, err_best, model_params_best


def main():
    # Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    utils.view_pc([pc])

    # Fit a plane to the data using ransac
    print("type of point cloud: ", type(pc))  # a list of 3D point
    print("Number of point: ", len(pc))      # 400 points
    # print(pc) np.matrix, [ [], [], [] ] 3x1

    # Show the resulting point cloud

    # Draw the fitted plane

    # inliers: red
    # outliers: blue
    # plane: green
    pc_size = len(pc)
    # threshold = 0.1
    threshold = 0.2
    ITER = 2000
    num_of_pts = 3
    N = pc_size * 0.6
    # N = pc_size * 0.25
    pt_in_plane, err_best, model_params_best = RANSAC(
        pc, num_of_pts, N, ITER, threshold)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print("ransac params: ", model_params_best)

    # plot the ori
    # final points. We use zip to get 1D lists of x, y and z coordinates.
    inliers = []
    outliers = []
    for pt in pc:
        if distance_to_plane(pt, model_params_best) < threshold:
            inliers.append(np.squeeze(np.asarray(pt)))
        else:
            outliers.append(np.squeeze(np.asarray(pt)))
    # print("inliers: ",inliers)
    for i in range(len(inliers)):
        ax.plot(*zip(inliers[i]), color='r',
                linestyle=' ', linewidth=1, marker='o')

    for i in range(len(outliers)):
        ax.plot(*zip(outliers[i]), color='b',
                linestyle=' ', linewidth=1, marker='o')

    xstart = -0.5
    xend = 1
    ystart = -0.25
    yend = 1.5
    ax.set_xlim([xstart, xend])
    ax.set_ylim([ystart, yend])

    a, b, c, d = model_params_best

    normal = np.matrix([a, b, c]).reshape(3, 1)
    pt = pt_in_plane
    color_green = (0.0, 1.0, 0.0, 0.5)
    ax.set(xlabel='x', ylabel='y', zlabel='z',
           title='Plane {0:.4f}x + {1:.4f}y + {2:.4f}z = {3:.4f}'.format(a, b, c, d))

    fig = utils.draw_plane(fig, normal, pt, color=color_green, length=[
                           xstart, xend], width=[ystart, yend])
    # print('The equation of the plane is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    plt.axis([-0.4, 1, -0.25, 1.5])

    # adjust the view so
    # we can see the point/plane alignment
    # ax.view_init(0, 22)
    plt.tight_layout()
    plt.savefig('ransac_plane.png')

    print("ransac ends!")
    ###YOUR CODE HERE###
    plt.show()

    input("Press enter to end:")


if __name__ == '__main__':
    main()
