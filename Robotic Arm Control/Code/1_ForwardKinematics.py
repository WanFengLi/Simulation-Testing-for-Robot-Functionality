
#This program is developed by Han Li

#This file try to control robot arm by forward kinematics
#and compare the calculated position with simulation position

#This file include head file as follows
#Please ensure necessary packages are available to be called before running
# -*- coding: utf-8 -*-
import sim
import time
import cv2
import numpy as np
import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, Symbol, symbols, solveset
from sympy import S, erf, log, sqrt, pi, sin, cos, tan
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

#Start Program and just in case, close all opened connections
print('Program started')
sim.simxFinish(-1)

#Define moving matrix
StepMatrix=[[0 for i in range(10)] for i in range(500)]
#first 6 parameters represent joint angle
#last 4 parameters control gripper fingers
#move joint 1,3,4,5,6 to desired position
StepMatrix[0]=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[1]=[0.2,0.0,-0.2,0.3,-0.7,-0.1,0.0,0.0,0.0,0.0]
StepMatrix[2]=[0.2,0.0,-0.2,0.3,-0.7,-0.1,0.0,0.0,0.0,0.0]
StepMatrix[3]=[0.2,0.0,-0.1236,0.3,-0.7,-0.1,0.0,0.0,0.0,0.0]
StepMatrix[4]=[0.0981,0.0,0.0,0.1472,-0.5180,-0.0491,0.0,0.0,0.0,0.0]
#move joint 2 to desired catching position
StepMatrix[5]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[6]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[7]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[8]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[9]=[0.0,-0.1963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#catch the object by gripper
StepMatrix[10]=[0.0,0.0,0.0,0.0,0.0,0.0,-0.01,0.01,0.01,-0.01]
StepMatrix[11]=[0.0,0.0,0.0,0.0,0.0,0.0,-0.01,0.01,0.01,-0.01]
StepMatrix[12]=[0.0,0.0,0.0,0.0,0.0,0.0,-0.01,0.01,0.01,-0.01]
#move joint 2 to lift the object
StepMatrix[13]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[14]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[15]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[16]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[17]=[0.0,0.1963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#move joint 1,3,5 to desired drop position
StepMatrix[18]=[-0.3,0.0,0.3,0.0,-0.3,0.0,0.0,0.0,0.0,0.0]
StepMatrix[19]=[-0.3,0.0,0.3,0.0,-0.3,0.0,0.0,0.0,0.0,0.0]
StepMatrix[20]=[-0.3,0.0,0.3,0.0,-0.3,0.0,0.0,0.0,0.0,0.0]
StepMatrix[21]=[-0.3,0.0,0.1472,0.0,-0.1472,0.0,0.0,0.0,0.0,0.0]
StepMatrix[22]=[-0.1962,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#move joint 2 to desired drop position
StepMatrix[23]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[24]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[25]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[26]=[0.0,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[27]=[0.0,-0.1963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#drop the object by gripper
StepMatrix[28]=[0.0,0.0,0.0,0.0,0.0,0.0,0.01,-0.01,-0.01,0.01]
StepMatrix[29]=[0.0,0.0,0.0,0.0,0.0,0.0,0.01,-0.01,-0.01,0.01]
StepMatrix[30]=[0.0,0.0,0.0,0.0,0.0,0.0,0.01,-0.01,-0.01,0.01]
#move joint 2 to leave the object
StepMatrix[31]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[32]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[33]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[34]=[0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[35]=[0.0,0.1963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#move joint to bring position back to original
StepMatrix[36]=[0.2,0.0,-0.2,-0.3,-0.7,0.1,0.0,0.0,0.0,0.0]
StepMatrix[37]=[0.2,0.0,-0.2,-0.3,-0.7,0.1,0.0,0.0,0.0,0.0]
StepMatrix[38]=[0.2,0.0,-0.1236,-0.3,-0.7,0.1,0.0,0.0,0.0,0.0]
StepMatrix[39]=[0.0981,0.0,0.0,-0.1472,-0.5180,0.0491,0.0,0.0,0.0,0.0]

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

		#Update control variables of angle and fingers
		joint1Angle += StepMatrix[counter][0]
		joint2Angle += StepMatrix[counter][1]
		joint3Angle += StepMatrix[counter][2]
		joint4Angle += StepMatrix[counter][3]
		joint5Angle += StepMatrix[counter][4]
		joint6Angle += StepMatrix[counter][5]
		finger1Move += StepMatrix[counter][6]
		finger2Move += StepMatrix[counter][7]
		finger3Move += StepMatrix[counter][8]
		finger4Move += StepMatrix[counter][9]

		#Correspond to the angle of each joint of the robotic arm
		#according to the initial value
		theta1=joint1Angle-pi/2
		theta2=-joint2Angle
		theta3=joint3Angle
		theta4=-joint4Angle+pi/2
		theta5=joint5Angle
		theta6=joint6Angle-pi/2
		theta7=joint7Angle
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
        #show end actuator position to compare with simulation results
		print(p7)

		#Set actuators on mobile robot
		sim.simxSetJointTargetPosition(clientID, joint1, joint1Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint2, joint2Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint3, joint3Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint4, joint4Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint5, joint5Angle, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetPosition(clientID, joint6, joint6Angle, sim.simx_opmode_oneshot)
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
