#!/usr/bin/env python
import utils
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from copy import deepcopy
###YOUR IMPORTS HERE###


def Principle_Component_Analysis(pc):
    mean = sum(pc)/len(pc)
    # to [0][0], [1][0], [2][0] : The first point x,y,c
    pc = utils.convert_pc_to_matrix(pc)
    size = pc.shape[1]  # 200 since shape=(3,200)
    # 2. Subtract X by mean
    pc = pc - mean
    # 4.
    cov_Q = pc * pc.T / (size-1)
    # 5. U, sigma, VT = SVD(Q)
    U, sigma, VT = np.linalg.svd(cov_Q)

    UT = np.transpose(U)
    V = np.transpose(VT)

    # 6. Xnew = VT * X
    pc_new = VT * pc

    # Show the resulting point cloud
    print("First V.T: ", VT)
    pc_after_rotation = utils.convert_matrix_to_pc(pc_new)
    fig2 = utils.view_pc([pc_after_rotation])

    # Rotate the points to align with the XY plane AND eliminate the noise
    ########### (b) : Pick the k-th largest pricipal components #############
    # Compute Variance of sigma
    S = sigma * sigma
    threshold = 0.01

    V_for_c = V # (C) here

    # Vs = np.empty([1, V.shape[1]])
    print("V shape: ",V.shape)
    print("V ",V)
    Vs = []

    for i in range(V.shape[1]):  # Loop through each column of V
        # print(i)
        if sigma[i] >= threshold:
            # Append V.column[i] == V[:,i]
            print("Append column ", i, " of V: ", V[:, i])
            Vs.append(np.asarray(V[:, i]).reshape(3, 1))
        else:
            # append column vector
            Vs.append(np.array([0, 0, 0]).reshape(3, 1))
            # Vs.append(np.array([0, 0, 0]))
        # Vs now: 3*n (n=2) matrix. Lose one dimension!
        # With the last col = [0 0 0]'
    # print("Vs before: ", Vs)

    Vs = np.asarray(Vs)

    print("Vs.shape: ",Vs.shape)
    print("Type(Vs) ", type(Vs))
    print("Final Vs: ", Vs)

    # Note : this Vs is already VsT
    print("Final VsT: ",np.squeeze(Vs))
    print("Vs.shape ", np.squeeze(Vs).shape)
    # print("Final Vs: ", Vs)

    # Vs.T: the last ROW = [0 0 0]
    VsT = np.transpose(Vs)
    pc_b = np.asmatrix(Vs) * pc 

    print("shape of pc_b: ", pc_b.shape)  # Should be 3x200
    # If dimension reduction : that axis value should be zero.
    pc_b = utils.convert_matrix_to_pc(pc_b)

    return pc_b, V_for_c


def main():

    # Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    fig = utils.view_pc([pc])

    pc_b, V_for_c = Principle_Component_Analysis(pc)

    # Show the resulting point cloud

    # print("V for c: ", V_for_c)
    # print("Vshape ", V_for_c.shape)

    # v1 = np.asarray(V_for_c[:, 0])
    # vec1 = np.ones([])

    pc_c = deepcopy(pc)
    fig3 = utils.view_pc([pc_c])
    v1 = []
    v2 = []
    for j in range(3):
        v1.append(V_for_c[j,0])
        v2.append(V_for_c[j,1])

    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # v1 = V_for_c[0]
    # v2 = V_for_c[1]
    print("v1 ", v1)
    print("v2 ", v2)
    normal = np.matrix(np.cross(v1, v2)).reshape(3,1)
    print("normal: ", normal)
    pt = pc_c[len(pc_c)/2]
    color_green = (0, 1.0, 0, 0.3)
    utils.draw_plane(fig3, normal, pt, color_green)
    # plt.savefig("pca_(c).png")
    
    # print(v1)
    # print(v2)
    # print(v2.shape)
    # normal = np.matrix([0, 0, 1]).reshape(3, 1)
    # pt = pc_b[0]
    # pt = np.array([0, 0, 0])

    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
