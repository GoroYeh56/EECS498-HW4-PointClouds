#!/usr/bin/env python
from copy import deepcopy
import utils
import numpy as np
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###


import pca_template as pca
import ransac_template as ransac
from ransac_template import distance_to_plane
from ransac_template import model_fit
import math
import random as rd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



###YOUR IMPORTS HERE###

def Error( pts, model_param):
    #TODO: sum of squared residuals from p to plane(model_param)
    err = 0
    for pt in pts:
        err +=  pow(distance_to_plane(pt, model_param) ,2)
    return err

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
        model_params = [a,b,c,d]
        return model_params


def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    fig = None

    errors_pca = []
    errors_ransac = []
    time_pca = []
    time_ransac = []
    iter = []
    NumOfOutliers = []

    for it in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        fig = utils.view_pc([pc])

        ###YOUR CODE HERE###
        print("\n============ Iteration : ",it ," =====================\n")
        iter.append(it+1)
        NumOfOutliers.append( (it+1)*10)
        ##### 1. pca : plane(green), inliers(red), outliers(blue)
        pca_threshold = 0.2
        pca_starttime =time.time()
        pc_after_pca, V_for_c = pca.Principle_Component_Analysis(pc)
        pca_endtime = time.time()
        
        # pc_c = deepcopy(pc)
        # fig3 = utils.view_pc([pc_c])

        # utils.draw_plane(fig, normal, pt, color_green)
        # Draw plane
        v1 = []
        v2 = []
        for j in range(3):
            v1.append(V_for_c[j,0])
            v2.append(V_for_c[j,1])

        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        normal = np.cross(v1, v2)
        # pt = pc[len(pc)/2]
        # p3 : mean point               
        p3 = sum(pc)/len(pc)
        mean_p = []

        for i in range(3):
            mean_p.append(p3[i].item(0))
            print("p3[i]", p3[i], type(p3[i]), p3[i].shape)
        mean_p = np.asarray(mean_p)
        print("mean point(p3): " , mean_p)
        a, b, c = normal
        d = np.dot(normal, mean_p)
        pca_plane_params = [a, b, c, d]
        print("a, b, c, d", a, b, c ,d)
        print("pca plane params: ", pca_plane_params)

        normal = np.matrix(np.cross(v1, v2)).reshape(3,1)
        print("normal: ", normal)
        # Draw pca plane & inliers & outliers
        # fig1 = utils.view_pc( [pc_after_pca] )
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(111, projection='3d')

        # pca_plane_params = np.array([0,0,1,0])
        # pca_plane_params = model_fit()
        # err_pca = Error(pc_after_pca, pca_plane_params) # TODO sum of squared error
        inliers = []
        outliers = []

        # pc_after_pca = deepcopy(pc)
        for pt in pc:
            if distance_to_plane(pt, pca_plane_params) < pca_threshold:
                inliers.append( np.squeeze(np.asarray(pt)))
            else:
                outliers.append( np.squeeze(np.asarray(pt)))

        for i in range(len(inliers)):
            ax.plot(*zip(inliers[i]), color='r', linestyle=' ', linewidth=1, marker='o')
        for i in range(len(outliers)):
            ax.plot(*zip(outliers[i]), color='b', linestyle=' ',linewidth =1, marker='o')
        err_pca = Error(inliers, pca_plane_params) # TODO sum of squared error

        # normal = np.matrix([0, 0, 1]).reshape(3,1)
        # pt = pc_after_pca[0]
        # pt = np.array([0, 0, 0])

        color_green = (0, 1.0, 0, 0.3)
        utils.draw_plane(fig1, normal , pt, color_green)
        ax.set(xlabel='x', ylabel='y', zlabel='z', title='Plane {0:.4f}x + {1:.4f}y + {2:.4f}z = {3:.4f}'.format(a,b,c,d))
        plt.savefig('pca_vs_ransac(a)_pca.png')

        if it== (num_tests-1):
            print("plt.show()...")
            plt.show()
        ###### 2. ransac : plane(green), inliers(red), outliers(blue)

        # fig = utils.view_pc([pc])
        pc_size = len(pc)
        threshold = 0.2
        ITER = 2000
        num_of_pts = 3
        N = pc_size * 0.6
        ransac_starttime = time.time()
        pt_in_plane, err_best, model_params_best = ransac.RANSAC(pc, num_of_pts, N, ITER, threshold)
        ransac_endtime = time.time()

        # Draw ransac plane & inliers & outliers
        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')

        print("size pc:" , len(pc))
        print("model params: ",model_params_best)
        inliers = []
        outliers = []
        for pt in pc:
            # print("pt: ",pt," dis: ", distance_to_plane(pt, model_params_best))
            if distance_to_plane(pt, model_params_best) < threshold:
                inliers.append(np.squeeze(np.asarray(pt)))
            else:
                outliers.append(np.squeeze(np.asarray(pt)))

        for i in range(len(inliers)):
            ax.plot(*zip(inliers[i]), color='r',
                    linestyle=' ', linewidth=1, marker='o')

        for i in range(len(outliers)):
            ax.plot(*zip(outliers[i]), color='b',
                    linestyle=' ', linewidth=1, marker='o')

        print("ransac profiling.....")
        print("inliers: ", len(inliers))
        print("outliers: ", len(outliers))

        xstart = -0.5
        xend = 1
        ystart = -0.25
        yend = 1.5
        ax.set_xlim([ xstart, xend])
        ax.set_ylim([ ystart, yend])
        a, b, c, d = model_params_best
        normal = np.matrix([a, b, c]).reshape(3,1)
        pt = pt_in_plane
        color_green = (0.0, 1.0, 0.0, 0.5)
        ax.set(xlabel='x', ylabel='y', zlabel='z', title='Plane {0:.4f}x + {1:.4f}y + {2:.4f}z = {3:.4f}'.format(a,b,c,d))
        fig = utils.draw_plane(fig, normal, pt_in_plane, color=color_green, length=[xstart, xend], width=[ystart, yend])
        plt.savefig('pca_vs_ransac(a)_ransac.png')

        if it== (num_tests -1):
            print("plt.show()...")
            plt.show()
            
        # 3. Error v.s. Number of Outliers (pca / ransac)
        errors_pca.append( np.squeeze(np.asarray(err_pca)).reshape(1) )
        errors_ransac.append( np.squeeze( np.asarray(err_best)).reshape(1) )
        # print("type err: ", type(err_best))
        # print("err_best.shape ",err_best.shape)

        # 4. Computation Time at each iteration (pca / ransac)
        time_pca.append( pca_endtime - pca_starttime)
        time_ransac.append( ransac_endtime - ransac_starttime)

        #this code is just for viewing, you can remove or change it
        # input("Press enter for next test:")
        if it< (num_tests-1):
            plt.close(fig1)
            plt.close(fig)
        else:
            print("last iteration, change view and save figure!")
            plt.close(fig1)
            plt.close(fig)
        ###YOUR CODE HERE###


    print("Number of outliers: ",NumOfOutliers)
    print("Error pca ", errors_pca)
    print("Error ransac ", errors_ransac)


    # print Error v.s. Number of Outliers
    plt.figure(3)
    plt.title("Error vs. Number of Outliers - PCA")
    plt.plot(NumOfOutliers, errors_pca, color = 'r')

    plt.xlabel("Number of Outlier")
    plt.ylabel("Error (sum of squared error)")
    plt.savefig('Error_vs_NumOutliers_PCA.png')

    plt.figure(4)
    plt.title("Error vs. Number of Outliers - RANSAC")
    plt.plot(NumOfOutliers, errors_ransac, color='g')
    plt.xlabel("Number of Outlier")
    plt.ylabel("Error (sum of squared error)")
    plt.savefig('Error_vs_NumOutliers_RANSAC.png')


    plt.figure(5)
    plt.title("Computation Time vs. Iteration - PCA")
    plt.plot(iter, time_pca, color = 'r')
    plt.xlabel("iteration #")
    plt.ylabel("Computation time(sec)")
    plt.savefig('ComputationTimevsIter_PCA.png')


    plt.figure(6)
    plt.title("Computation Time vs. Iteration - RANSAC")
    plt.plot(iter, time_ransac, color='g')
    plt.xlabel("iteration #")
    plt.ylabel("Computation time(sec)")
    plt.savefig('ComputationTimevsIter_RANSAC.png')

    print("\n ...... End pca_vs_ransac.py......")

    plt.show()
    input("Press enter to end")


if __name__ == '__main__':
    main()
