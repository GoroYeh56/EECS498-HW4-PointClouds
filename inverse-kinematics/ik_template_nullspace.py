import numpy as np
from pybullet_tools.utils import connect, disconnect, set_joint_positions, wait_if_gui, set_point, load_model,\
                                 joint_from_name, link_from_name, get_joint_info, HideOutput, get_com_pose, wait_for_duration
from pybullet_tools.transformations import quaternion_matrix
from pybullet_tools.pr2_utils import DRAKE_PR2_URDF
import time
import sys
### YOUR IMPORTS HERE ###

import math
from math import fabs
from numpy.linalg import norm
#########################

from utils import draw_sphere_marker

def get_ee_transform(robot, joint_indices, joint_vals=None):
    # returns end-effector transform in the world frame with input joint configuration or with current configuration if not specified
    if joint_vals is not None:
        set_joint_positions(robot, joint_indices, joint_vals)
    ee_link = 'l_gripper_tool_frame'
    pos, orn = get_com_pose(robot, link_from_name(robot, ee_link))
    res = quaternion_matrix(orn)
    res[:3, 3] = pos
    return res

def get_joint_axis(robot, joint_idx):
    # returns joint axis in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    R_W_J = H_W_J[:3, :3]
    joint_axis_local = np.array(j_info.jointAxis)
    joint_axis_world = np.dot(R_W_J, joint_axis_local)
    return joint_axis_world

def get_joint_position(robot, joint_idx):
    # returns joint position in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    j_world_posi = H_W_J[:3, 3]
    return j_world_posi

def set_joint_positions_np(robot, joints, q_arr):
    # set active DOF values from a numpy array
    q = [q_arr[0, i] for i in range(q_arr.shape[1])]
    set_joint_positions(robot, joints, q)


def get_translation_jacobian(robot, joint_indices, current_ee_position):
    J = np.zeros((3, len(joint_indices)))
    ### YOUR CODE HERE ###
    # Use ONLY Position! SO
    # 3x7 matrix
    # dx/djoint1  dx/djoint2 ... (x : end-effector x)
    # dy/djoint1      dy/dj2     ...
    # dz/djoint1      dz/dj1     ...
    # x, y, z = end-effector's position
    # 
    print(joint_indices)
    P = np.zeros((3, len(joint_indices)))
    for i in range(len(joint_indices)):
        p =  current_ee_position - get_joint_position(robot, joint_indices[i])
        J[:,i] = np.cross(get_joint_axis(robot, joint_indices[i]), p)
    ### YOUR CODE HERE ###
    return J

def get_translation_jacobian2(robot, joint_indices, current_ee_position):
    J = np.zeros((3, len(joint_indices)))
    ### YOUR CODE HERE ###

    P = np.zeros((3, len(joint_indices)))
    
    # print("joint_indices: ",joint_indices)

    for i in range(len(joint_indices)):
        # print("p[",i,"] : ",get_joint_position(robot, joint_indices[i]))
        # print("cur: ", current_ee_position)
        p = current_ee_position - get_joint_position(robot, joint_indices[i]) 
        # print(p)
        P[:,i] = p
        J[:,i] = np.cross(get_joint_axis(robot, joint_indices[i]), p)
    # print("Jacobian: ",J)
    # print("P: ",P)
    ### YOUR CODE HERE ###
    return J


def get_jacobian_pinv(J):
    J_pinv = []
    ### YOUR CODE HERE ###

    # J_pinv = np.linalg.pinv(J)
    lambda_var = 0.1 # TODO be tuned for damped least square
    # print(type(J))
    # print(J.shape)
    eye = np.identity( (J @ np.transpose(J)).shape[0] )
    J_pinv = np.transpose(J) @  np.linalg.inv(J @ np.transpose(J) + lambda_var*lambda_var*eye )

    # print("J_pinv shape: ",J_pinv.shape)

    ### YOUR CODE HERE ###
    return J_pinv

def tuck_arm(robot):
    joint_names = ['torso_lift_joint','l_shoulder_lift_joint','l_elbow_flex_joint',\
        'l_wrist_flex_joint','r_shoulder_lift_joint','r_elbow_flex_joint','r_wrist_flex_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    set_joint_positions(robot, joint_idx, (0.24,1.29023451,-2.32099996,-0.69800004,1.27843491,-2.32100002,-0.69799996))

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Specify which target to run:")
        print("  'python3 ik_template.py [target index]' will run the simulation for a specific target index (0-4)")
        exit()
    test_idx = 0
    try:
        test_idx = int(args[0])
    except:
        print("ERROR: Test index has not been specified")
        exit()

    # initialize PyBullet
    connect(use_gui=True, shadows=False)
    # load robot
    with HideOutput():
        robot = load_model(DRAKE_PR2_URDF, fixed_base=True)
        set_point(robot, (-0.75, -0.07551, 0.02))
    tuck_arm(robot)
    # define active DoFs
    joint_names =['l_shoulder_pan_joint','l_shoulder_lift_joint','l_upper_arm_roll_joint', \
        'l_elbow_flex_joint','l_forearm_roll_joint','l_wrist_flex_joint','l_wrist_roll_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    # intial config
    q_arr = np.zeros((1, len(joint_idx)))
    set_joint_positions_np(robot, joint_idx, q_arr)
    # list of example targets
    targets = [[-0.15070158,  0.47726995, 1.56714123],
               [-0.36535318,  0.11249,    1.08326675],
               [-0.56491217,  0.011443,   1.2922572 ],
               [-1.07012697,  0.81909669, 0.47344636],
               [-1.11050811,  0.97000718,  1.31087581]]
    # define joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robot, joint_idx[i]).jointLowerLimit, get_joint_info(robot, joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}
    q = np.zeros((1, len(joint_names))) # start at this configuration
    target = targets[test_idx]
    # draw a blue sphere at the target
    draw_sphere_marker(target, 0.05, (0, 0, 1, 1))
    
    ### YOUR CODE HERE ###

    # 5 targets, test which target

    # joint_values = []
    # for i in range(len(joint_idx)):        
    #     print("joint ", i, " axis: ", get_joint_axis(robot, i))
    #     joint_value = get_joint_position(robot, i) # cartesion position
    #     print("joint_value: ",joint_value)
    #     joint_values.append(joint_value)

    ### get_joint_axis(robot, i) : Return joint i's axis at current configuration 
    ### robot_arm's current arm's pose (every joint configuration)

    # print("end-effector transform: ", get_ee_transform(robot, joint_idx, joint_values))

    # print("J pinv ", J_pinv)

    # for i in range(3):
    #     joint_world_pos = get_joint_position(robot, joint_indices)
    #     joint_axis_world = get_joint_axis(robot, joint_indices)
    #     res = get_ee_transform(robot, joint_indices, joint_world_pos)

    print("Joint limits: ",joint_limits)

    # Note: configuration is the joint_values for each joint (motors' angles)
    def FK( robot, joint_idx, configuration ):
        HTM = get_ee_transform(robot, joint_idx, configuration) # Homogeneous Transformation matrix
        
        end_efffector_position = HTM[:3, 3]
        return np.squeeze( end_efffector_position.reshape(1,3))
        # print("FK: ")
        # print("HTM: ",HTM)
        # print(type(end_efffector_position))
        # print(end_efffector_position.shape)
        # print(np.squeeze(HTM[:3, 3].reshape(1,3)))

    # Use Euclidean 2 norm here
    # def norm(x):
        # return math.sqrt(x * x)
                                    # joint_idx is a list of number
    # Map joint_limits to JL[idx] => limit[0], [1]
    JL = [[0,0]]*len(joint_limits)
    for i in range(len(joint_limits)):
        JL[i] = list(joint_limits[joint_names[i]])
    # print("JL ",JL)
    print("joint_idx: ",joint_idx)

    # TODO : Repel from joint limits
    def get_q2dot():

        pass

    def Iterative_Inverse_Kinematics(robot, joint_idx, joint_limits, qint, xtarget, threshold = 0.1, alpha=0.05,MAX_ITER=100, beta=1, stepsize=0.5):
        """
        @ x : a vetor of end-effector's position [x y z]'
        @ 
        
        """
        qcur = qint

        it = 0
        while True:
            it = it+1
            if it>MAX_ITER:
                break
            xcur = FK(robot, joint_idx, qcur) # current end-effector position
            # print("xcur after FK: ",xcur)
            xdot = (xtarget - xcur) #
            # print("type xtarget, xcur, xdot ", type(xtarget), " ",type(xcur)," ", type(xdot))
            # print("xdot: ",xdot)

            error = norm(xdot)
            print("\n======= Iteration: ", it, " ===========")
            print("error: ", error, " threshold: ", threshold)
            # print("qcur: ",qcur)
            # print("xcur: ",xcur)
            if error < threshold:
                return qcur

            # J = get_translation_jacobian(robot, joint_idx)
            J = get_translation_jacobian(robot, joint_idx, xcur)
            print("J ",J)

            # (d) TODO: Here Modify dq/dt to Secondary Task Format
            # Secondary Task : Repel from joint limit

            q2dot = np.array([0, 0, 0, 0, 0, 0, 0])
            # For each joint:
            epsilon = 0.01
            for i in range(len(joint_idx)):
                # print("qnew[i]: ",qnew[i])
                # print("joint_limits[i] ", joint_limits[i])
                if( fabs(joint_limits[i][1] - qcur[i]) >= fabs( joint_limits[i][0] - qcur[i]) ):
                    q2dot[i] = stepsize * (1/ (fabs(joint_limits[i][0] - qcur[i]) + epsilon) )
                else:
                    q2dot[i] = - stepsize * (1/ (fabs(joint_limits[i][1] - qcur[i])+epsilon) )

            I = np.identity( (get_jacobian_pinv(J) @ J).shape[0])
            print(I.shape)
            qdot = get_jacobian_pinv(J)@xdot  + beta * ( I - get_jacobian_pinv(J) @ J) @ q2dot.T

            # qdot = get_jacobian_pinv(J) @ xdot
            
            if norm(qdot) > alpha:
                qdot = alpha * (qdot / norm(qdot))

            qnew = qcur + qdot
            #print("qnew : ",qnew)
            
            # handle joint limit  a dictionary
            for i in range(len(joint_idx)):
                # print("qnew[i]: ",qnew[i])
                # print("joint_limits[i] ", joint_limits[i])
                if i==4 or i==6:
                    continue
                if qnew[i] > joint_limits[i][1]:
                    print("joint ", i, " hit upper limit")
                    qnew[i] = joint_limits[i][1]
                elif qnew[i] < joint_limits[i][0]:
                    print("joint ",i , " hit lower limit")
                    qnew[i] = joint_limits[i][0]
                else:
                    pass # else do nothine
            qcur = qnew
            
        return qcur

    # (d)
    qint = np.array([0, 0, 0, 0, 0, 0, 0])
    xtarget = target
    alpha = 0.05   # step-size: scale of qdot (joint angle moving range at each iteration)
    MAX_ITER = 1000
    threshold = 0.035
    
    # Secondary task Parameters to be tuned
    beta = 1
    stepsize = 0.5

    print("Start config:",qint)
    print("Target ee position: ",xtarget) # a list of position x,y,z
    qfinal = Iterative_Inverse_Kinematics(robot, joint_idx, JL, qint, xtarget, threshold, alpha, MAX_ITER, beta, stepsize)
    print("qfinal: ", qfinal)


    # (d)






    ### YOUR CODE HERE ###

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()