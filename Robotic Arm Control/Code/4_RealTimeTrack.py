
#This program is developed by Han Li

#This file try to make the end effector hover over the block in real time

#This file include head file as follows
#Please ensure necessary packages are available to be called before running
# -*- coding: utf-8 -*-
import sim
import time
import cv2
import cv2 as cv
import numpy as np
import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, Symbol, symbols, solveset, solve, simplify
from sympy import S, erf, log, sqrt, pi, sin, cos, tan, diff, det
from sympy import init_printing

#Define transform matrices for roll, yaw and pitch
def T(x, y, z):
    T_xyz = Matrix([[1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])
    return T_xyz
def Rx(roll):
    R_x = Matrix([[1, 0, 0, 0],
                  [0, cos(roll), -sin(roll), 0],
                  [0, sin(roll), cos(roll), 0],
                  [0, 0, 0, 1]])
    return R_x
def Ry(pitch):
    R_y = Matrix([[ cos(pitch), 0, sin(pitch), 0],
                  [ 0, 1, 0, 0],
                  [-sin(pitch), 0, cos(pitch), 0],
                  [ 0, 0, 0, 1]])
    return R_y
def Rz(yaw):
    R_z = Matrix([[cos(yaw),-sin(yaw), 0, 0],
                  [sin(yaw), cos(yaw), 0, 0],
                  [ 0, 0, 1, 0],
                  [ 0, 0, 0, 1]])
    return R_z

#Simulation variables initialization
joint1Angle = 0.0
joint2Angle = 0.0
joint3Angle = 0.0
joint4Angle = 0.0
joint5Angle = 0.0
joint6Angle = 0.0
joint7Angle = 0.0
finger1Move = 0.0
finger2Move = 0.0
finger3Move = 0.0
finger4Move = 0.0
#initialize error value
t = 0.0

#Start Program and just in case, close all opened connections
print('Program started')
sim.simxFinish(-1)

#set a counter to count for running step
counter = 0

#Connect to simulator running on localhost
#V-REP runs on port 19997, a script opens the API on port 19999
clientID = sim.simxStart('10.181.132.129', 19999, True, True, 5000, 5)
thistime = time.time()

#Connect to the simulation
if clientID != -1:
	print('Connected to remote API server')
	#Get handles to simulation objects
	print('Obtaining handles of simulation objects')
	#Floor
	res,floor = sim.simxGetObjectHandle(clientID, 'ResizableFloor_5_25', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to Floor')
	#Robot Arm Base (reference point)
	res,arm = sim.simxGetObjectHandle(clientID, 'redundantRobot', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to Robot')
	#End Effector camera for visual servoing
	res,camera = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to Camera')
	#Joint 1
	res,joint1 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint1', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to redundantRob_joint1')
	#Joint 2
	res,joint2 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint2', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to redundantRob_joint2')
	#Joint 3
	res,joint3 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint3', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to redundantRob_joint3')
	#Joint 4
	res,joint4 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint4', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to redundantRob_joint4')
	#Joint 5
	res,joint5 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint5', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to redundantRob_joint5')
	#Joint 6
	res,joint6 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint6', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to redundantRob_joint6')
    #Joint 7
	res,joint7 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint7', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to redundantRob_joint7')
	#Finger1
	res,finger1 = sim.simxGetObjectHandle(clientID, 'Finger_joint1', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to Finger_joint1')
	#Finger2
	res,finger2 = sim.simxGetObjectHandle(clientID, 'Finger_joint2', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to Finger_joint2')
    #Finger3
	res,finger3 = sim.simxGetObjectHandle(clientID, 'Finger_joint3', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to Finger_joint3')
    #Finger4
	res,finger4 = sim.simxGetObjectHandle(clientID, 'Finger_joint4', sim.simx_opmode_oneshot_wait)
	if res != sim.simx_return_ok: print('Could not get handle to Finger_joint4')

    #Define the initial joint angle of the robotic arm
	theta_i = Matrix([-pi/2,0,0,pi/2,0,pi/2,0])
    #Define the initial end effector position
	POriginal = Matrix([0.32400,0.0,0.36048])

	#Start main control loop
	print('Starting control loop')
	res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_streaming)
	while (sim.simxGetConnectionId(clientID) != -1):
		#Get image from Camera
		lasttime = thistime
		thistime = time.time()
		res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_buffer)
		if res == sim.simx_return_ok:
			#Process image
			print("Image OK!", "{:02.1f}".format(1.0 / (thistime - lasttime)), "FPS")
			#Convert from V-REP flat RGB representation to OpenCV BGR colour planes
			original = np.array(image, dtype=np.uint8)
            #extract original image here for anothor use
			original1 = np.array(image, dtype=np.uint8)
			original.resize([resolution[0], resolution[1], 3])
			original = cv2.flip(original, 0)
			original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
			#Filter the image into components (alternately, cv2.split includes components of greys, etc.)
			blue = cv2.inRange(original, np.array([224,0,0]), np.array([255,32,32]))
			green = cv2.inRange(original, np.array([0,224,0]), np.array([32,255,32]))
			red = cv2.inRange(original, np.array([0,0,224]), np.array([32,32,255]))
			#Apply Canny edge detection
			blueEdges = cv2.Canny(blue, 32, 64)
			greenEdges = cv2.Canny(green, 32, 64)
			redEdges = cv2.Canny(red, 32, 64)
			#Combine edges from red, green, and blue channels
			edges = cv2.merge((blueEdges, greenEdges, redEdges))
			#Images must all be the same dimensions as reported by original.shape
			images = np.vstack((original, edges))
            #define the threshold for tracking
			ret,thresh = cv.threshold(original1,127,255,0)
            #extract information in image
			contours,hierarchy = cv.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			cnt = contours[0]
            #extract the pixels coordinate and side length
			x1,y1,w1,h1 = cv2.boundingRect(cnt)
            #use one of them as control variable
			print('pixels error is',y1)
            #convert it to simulation size
			t = (432459.0-y1)*0.0000005291
# 			if t < 0.0001:
# 				t=0
			print('t is', t)
			#Show processed images together in OpenCV window
			cv2.imshow('Camera', images)
			components = np.vstack((blue, green, red))
			cv2.imshow('Components', components)
		elif res == sim.simx_return_novalue_flag:
			#Camera has not started or is not returning images
			print("No image yet")
			pass
		else:
			#Something else has happened
			print("Unexpected error returned", res)

        #Symbolize angle values for inverse kinematics calculations
		theta1,theta2,theta3,theta4,theta5,theta6,theta7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
		theta = Matrix([theta1,theta2,theta3,theta4,theta5,theta6,theta7])
        ##Symbolize delta angle values for inverse kinematics calculations
		dtheta1,dtheta2,dtheta3,dtheta4,dtheta5,dtheta6,dtheta7 = symbols('dtheta_1 dtheta_2 dtheta_3 dtheta_4 dtheta_5 dtheta_6 dtheta_7')
		dtheta = Matrix([dtheta1,dtheta2,dtheta3,dtheta4,dtheta5,dtheta6,dtheta7])

		#Define the gripper offset in simulation
		gripper_offset=0.0743
		#Calculate the transformation matrix of each joint
		T1 = Ry(-pi/2) * T(0.0647, 0, 0) * Rx(theta1)
		T2 = T1 * T(0.13916, 0, 0) * Rz(theta2)
		T3 = T2 * T(0.09657, 0, 0) * Rx(theta3)
		T4 = T3 * T(0.19496, 0, 0) * Rz(theta4)
		T5 = T4 * T(0.09008, 0, 0) * Rx(theta5)
		T6 = T5 * T(0.23354, 0, 0) * Rz(theta6)
		T7 = T6 * T(0.06061, 0, 0) * Rx(theta7)*T(gripper_offset,0,0)
		#calculate the theoretical position of each joint by transformation matrix
		p0 = Matrix([0,0,0,1])
		p1 = T1 * p0
		p2 = T2 * p0
		p3 = T3 * p0
		p4 = T4 * p0
		p5 = T5 * p0
		p6 = T6 * p0
		p7 = T7 * p0

        #convert 4-dimensional homogeneous representation to a 3-dimensional vector
		p_i = Matrix([p7[0], p7[1], p7[2]])
        #produce the Jacobian
		J = p_i.jacobian(theta)
		Jsub = J.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		Jeval = Jsub.evalf()
        #use step matrix to control robot step by step
		dp = Matrix([0.0,t,0.0])
        #iterate the position value of end effector
        #set 3 joint to zero to gain a certain inverse solution
		dtheta=dtheta.subs({dtheta3:0,dtheta5:0,dtheta7:0})
        #solve joint 1,2,4 related to 6
		output=solve(Jeval*dtheta-dp,(dtheta1,dtheta2,dtheta4))
        #to make end effector downwards, the value of joint 2,4,6 should be 0
		ctrl=output[dtheta2]+output[dtheta4]+dtheta6
        #Solve value of dtheta6 with constraints
		ctrlOutput=solve(ctrl,dtheta6)
        #subtitute them to real number
		Dtheta6=ctrlOutput[0]
		Dtheta1=output[dtheta1].subs({dtheta6:Dtheta6})
		Dtheta2=output[dtheta2].subs({dtheta6:Dtheta6})
		Dtheta4=output[dtheta4].subs({dtheta6:Dtheta6})
        #Update joints with changed angles
		theta_i[0] += Dtheta1
		theta_i[1] += Dtheta2
		theta_i[3] += Dtheta4
		theta_i[5] += Dtheta6
		#use p0sub with real number to calculate Theoretical position value
		p0sub = p0.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		p1sub = p1.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		p2sub = p2.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		p3sub = p3.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		p4sub = p4.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		p5sub = p5.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		p6sub = p6.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
		p7sub = p7.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5], theta7:theta_i[6]})
        #Record the end effector position as the current value
		pCurrent= Matrix([p7sub[0],p7sub[1],p7sub[2]])
        #Update the angle of each joint
		joint1Angle += Dtheta1
		joint2Angle -= Dtheta2
		joint3Angle += 0
		joint4Angle -= Dtheta4
		joint5Angle += 0
		joint6Angle += Dtheta6
		joint7Angle += Dtheta1
		#Set actuators on mobile robot
		sim.simxSetJointTargetPosition(clientID, joint1, joint1Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint2, joint2Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint3, joint3Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint4, joint4Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint5, joint5Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint6, joint6Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint7, joint7Angle, sim.simx_opmode_oneshot)
        #show the end effector position for check
		print('p now is', pCurrent)
        #Set actuators on mobile robot
		sim.simxSetJointTargetPosition(clientID, finger1, finger1Move, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, finger2, finger2Move, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, finger3, finger3Move, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, finger4, finger4Move, sim.simx_opmode_oneshot)
        #change counter to go to next step
		counter += 1

	#End simulation
	sim.simxFinish(clientID)

else:
	print('Could not connect to remote API server')


#Close all simulation elements
sim.simxFinish(clientID)
cv2.destroyAllWindows()
print('Simulation ended')
