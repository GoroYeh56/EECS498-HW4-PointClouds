#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    target = 1
    if target == 0:
        pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target
    elif target==1:
        pc_target = utils.load_pc('cloud_icp_target1.csv') # Change this to load in a different target
    elif target==2:
        pc_target = utils.load_pc('cloud_icp_target2.csv') # Change this to load in a different target
    elif target==3:
        pc_target = utils.load_pc('cloud_icp_target3.csv') # Change this to load in a different target


    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])
    ### ICP algorithms here ###
    # p : a np.array, shape=(3,1)
    def Distance(p, q): 
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        dz = p[2] - q[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def findClosestPoint(p, pc_target):     
        dist = [Distance(p, point) for point in pc_target]
        # q = (pc_target[np.argmin(dist)]) 
        # print("closest point in pc_target: ", q)
        return (pc_target[np.argmin(dist)]) 

    def GetTransform(P, Q):

        R = 0 # Rotational matrix
        t = 0 # translation matrix
        pmean = sum(P)/len(P)
        qmean = sum(Q)/len(Q)

        X = P
        Y = Q
        X[:] = [x - pmean for x in X]
        Y[:] = [y - qmean for y in Y]

        X = np.reshape( np.array(X), (len(P),3))
        Y = np.reshape( np.array(Y), (len(P),3))
        
        S = np.matmul(np.transpose(X),Y)
        # S = np.matmul(X, np.transpose(Y))
        # print("S.shape ", S.shape)
        U, sigma, VT = np.linalg.svd(S)
        UT = np.transpose(U)
        V = np.transpose(VT)
        M = np.array([[1,0,0],[0,1,0], [0,0,np.linalg.det(V @ UT)]])

        # print("M shape", M.shape)
        # print("V shape", V.shape)
        # print("UT shape", UT.shape)

        R = np.matmul( np.matmul(V, M), UT)
        t = np.asarray(qmean - np.matmul(R,pmean) )
        
        return R, t



    def ComputeDistance(R, t, P, Q):
        # dist = np.matmul(R,P) + t - Q
        err = 0
        dist_vec = np.matmul(R,P)+t - Q 
        for i in range(len(P)):
            err += (dist_vec[i][0]*dist_vec[i][0] + \
                        dist_vec[i][1]*dist_vec[i][1] + \
                        dist_vec[i][2]*dist_vec[i][2])
        print("err: ",err)
        return err

    def ICP(pc_source, pc_target, epsilon=0.5, MAX_ITER=100):
        idx = 0
        it = []
        distances = []
        while True:
            
            idx = idx + 1
            it.append(idx)
            print("i: ",idx)
            if idx > MAX_ITER: 
                break

            P = [] # p1 -> q1 ,? =>  P[0]=> Q[0] 495 points
            Q = []
            for p in pc_source:
                q = findClosestPoint(p, pc_target)
                P.append(p)
                Q.append(q)

            R, t = GetTransform(P,Q)
            distance = ComputeDistance(R, t, pc_source, pc_target)
            distances.append(np.squeeze(distance).item(0))
            if distance <= epsilon:
                for i in range(len(P)):
                    pc_source[i] = np.matmul(R,pc_source[i]) + t                
                print("Small enough! Break...")
                break 
            # update all P
            for i in range(len(P)):
                pc_source[i] = np.matmul(R,pc_source[i]) + t

            print("\n======= Iteration ",idx," =========")
            print("R: ",R)
            print("det(R) ", np.linalg.det(R))
            print("t: ",t)
            print("pc_source[0]: ",pc_source[0])
        return pc_source, pc_target, it, distances

    # pc_source: list (len 495)
    # epsilon = 0.2
    epsilons = [0.061, 0.04, 0.06, 1.49]
    epsilon = epsilons[target]
    MAX_ITER = 100
    # epsilon = 0.04 # termination condition (0.35 might work)
    # C : correspondence: a dictionary.{}, key: pi => value: qi
    pc_source, pc_target, it, errors = ICP(pc_source, pc_target, epsilon, MAX_ITER)

    # plt.savefig('icp_'+str(target)+'_after.png')

    # utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    # plt.axis([-0.15, 0.15, -0.15, 0.15])

    # Plot Error v.s. iterations
    print(it)
    print(errors)
    # plot our list in X,Y coordinates
    
    plt.figure(2)
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.title("Error vs. Iteration of ICP")    
    plt.plot(it, errors, color='green')
    plt.savefig('icp_target_'+str(target)+'.png')
    ###YOUR CODE HERE###

    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])

    plt.show()
    # raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
