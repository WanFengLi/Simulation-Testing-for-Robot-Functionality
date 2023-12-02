
#This program is developed by Han Li

#This file try to pick up patient in random locations

#This file include head file as follows
#Please ensure necessary packages are available to be called before running
import sim
import time
import cv2
import numpy as np

#Simulation variables initialization
joint1Angle = 0.0
joint2Angle = 0.0
joint3Angle = 0.0
joint4Angle = 0.0
joint5Angle = 0.0
finger1Move = 0.0
finger2Move = 0.0
finger3Move = 0.0
finger4Move = 0.0

#Define moving matrix
StepMatrix=[[0 for i in range(10)] for i in range(500)]

#first 2 parameters represent desired movement for left and right wheel
#last 4 parameters control gripper fingers
#middle 3 parameters represent three joints of robotic arm
#wait for moveing at the beginning
StepMatrix[0]=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#move the last joint first to avoid collision
StepMatrix[3]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[4]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[5]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[6]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[7]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[8]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[9]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[10]=[0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0]
#Move the manipulator to the position to be graspe
StepMatrix[11]=[0.0,0.0,-0.3,0.0,0.25,0.0,0.0,0.0,0.0]
StepMatrix[14]=[0.0,0.0,-0.4,0.0,0.2,0.0,0.0,0.0,0.0]
StepMatrix[24]=[0.0,0.0,-0.2,0.0,0.2,0.0,0.0,0.0,0.0]
StepMatrix[25]=[0.0,0.0,0.0,0.2,0.0,0.0,0.0,0.0,0.0]
StepMatrix[26]=[0.0,0.0,-0.45,0.0,0.0,0.0,0.0,0.0,0.0]

#move forward to the position to grab
StepMatrix[27]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[29]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[31]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[33]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[35]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[37]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[39]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[41]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[43]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[45]=[-0.3,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[47]=[-0.4,-0.4,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[49]=[-0.4,-0.4,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[53]=[-0.5,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[55]=[-0.5,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[57]=[-0.5,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[59]=[-0.5,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

#control finger to grab Mr.York
StepMatrix[61]=[0.0,0.0,0.0,0.0,0.0,-0.02,0.02,0.02,-0.02]
StepMatrix[62]=[0.0,0.0,0.0,0.0,0.0,-0.02,0.02,0.02,-0.02]
StepMatrix[63]=[0.0,0.0,0.0,0.0,0.0,-0.02,0.02,0.02,-0.02]
StepMatrix[64]=[0.0,0.0,0.0,0.0,0.0,-0.02,0.02,0.02,-0.02]

#lift Mr.York
StepMatrix[70]=[0.0,0.0,0.4,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[73]=[0.0,0.0,0.0,-0.2,0.0,0.0,0.0,0.0,0.0]
StepMatrix[76]=[0.0,0.0,0.3,0.0,-0.1,0.0,0.0,0.0,0.0]
StepMatrix[79]=[0.0,0.0,0.4,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[82]=[0.0,0.0,0.4,0.0,0.0,0.0,0.0,0.0,0.0]
StepMatrix[85]=[0.0,0.0,0.0,0.0,-0.1,0.0,0.0,0.0,0.0]

#set a counter to count for operation step
counter = 0
#set a variable to change the state
state = 1
#initialize the variable for OpenCV functions
centerX = 0
midX = 0

#Start Program and just in case, close all opened connections
print('Program started')
sim.simxFinish(-1)

#Connect to simulator running on localhost
#V-REP runs on port 19997, a script opens the API on port 19999
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
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
    res,arm = sim.simxGetObjectHandle(clientID, 'Mobile_robot', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Robot')
    #End Effector camera for visual servoing
    res,camera = sim.simxGetObjectHandle(clientID, 'Vision_sensor_Front', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Camera')
    #Joint 1
    res,joint1 = sim.simxGetObjectHandle(clientID, 'Left_motor', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 1
    res,joint2 = sim.simxGetObjectHandle(clientID, 'Right_motor', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 1
    res,joint3 = sim.simxGetObjectHandle(clientID, 'Revolute_joint1', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 1
    res,joint4 = sim.simxGetObjectHandle(clientID, 'Revolute_joint2', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 1
    res,joint5 = sim.simxGetObjectHandle(clientID, 'Revolute_joint3', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    res,finger1 = sim.simxGetObjectHandle(clientID, 'fingermotor1', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Finger_joint1')
    #Finger2
    res,finger2 = sim.simxGetObjectHandle(clientID, 'fingermotor2', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Finger_joint2')
    #Finger3
    res,finger3 = sim.simxGetObjectHandle(clientID, 'fingermotor3', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Finger_joint3')
    #Finger4
    res,finger4 = sim.simxGetObjectHandle(clientID, 'fingermotor4', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Finger_joint4')

    #Start main control loop
    print('Starting control loop')
    res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_streaming)
    while (sim.simxGetConnectionId(clientID) != -1):
        
        #Get image from Camera
        lasttime = thistime
        thistime = time.time()
        if thistime - lasttime == 0:
            print('Divición entre cero')
            
        res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_buffer)
        if res == sim.simx_return_ok:
            #Process image 
            print("Image OK!", "{:02.1f}".format(1.0 / (thistime - lasttime)), "FPS")
            #Convert from V-REP flat RGB representation to OpenCV BGR colour planes
            original = np.array(image, dtype=np.uint8)
            #ima = np.array(image2, dtype=np.uint8)
            original.resize([resolution[0], resolution[1], 3])
            original = cv2.flip(original, 0)
            original = cv2.rotate(original, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)#COLOR_RGB2BGR
            #original2 = cv2.cvtColor(ima, cv2.COLOR_RGB2GRAY)#COLOR_RGB2BGR
            #dist = cv2.absdiff(original,original)
            
            #Detect the rows and colums
            rows, cols, _ = original.shape
            #print("\nrows:", rows, " cols:", cols)
            
            #this give us the number of pixels in the row and colums, devides by 2 because we need the middle
            centerX=int(rows/2)
            centerY=int(cols/2)
            midX=int(rows/2)
            midY=int(cols/2)           
            #Filter the image into components (alternately, cv2.split includes components of greys, etc.)
            blue = cv2.inRange(original, np.array([224,0,0]), np.array([255,32,32]))
            blue_low = np.array([120,50,50],dtype=np.uint8)
            blue_up = np.array([180,255,255],dtype=np.uint8)
            
            green = cv2.inRange(original, np.array([0,224,0]), np.array([32,255,32]))
            green_low = np.array([50,50,50],dtype=np.uint8)
            green_up = np.array([80,255,255],dtype=np.uint8)
            
            red = cv2.inRange(original, np.array([0,0,224]), np.array([32,32,255]))
            red_low = np.array([0,50,50],dtype=np.uint8)
            red_up = np.array([10,255,255],dtype=np.uint8)
            
            #Apply Canny edge detection
            blueEdges = cv2.Canny(blue, 32, 64)
            greenEdges = cv2.Canny(green, 32, 64)
            redEdges = cv2.Canny(red, 32, 64)
            #Combine edges from red, green, and blue channels
            edges = cv2.merge((blueEdges, greenEdges, redEdges))
            #Images must all be the same dimensions as reported by original.shape 
            images = np.vstack((original, edges))
            
            #Convert to HSV
            hsv=cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
            imgBin = cv2.inRange(hsv, green_low, green_up)

            
            #Detect the contours
            cnt, hie = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            #This state makes robot approach Mr.York until a certain distance
            for contours in cnt:
                print("Contour Area now is: ",cv2.contourArea(contours))
                if state == 2:
                    if cv2.contourArea(contours)<=18000 and cv2.contourArea(contours)>5000:
                        #make to wheels running slowly
                        joint1Angle -= 0.2
                        joint2Angle -= 0.2
                        print('\nNow in state 2')
                    if cv2.contourArea(contours)<=5000 and cv2.contourArea(contours)>0.0:
                        #make to wheels running slowly
                        joint1Angle -= 0.05
                        joint2Angle -= 0.05
                        print('\nNow in state 2')
                    elif cv2.contourArea(contours)>18000:
                        #set wheels 0 to prepare for state 3
                        joint1Angle -= 0
                        joint2Angle -= 0
                        print('\nNow in state 2')
                        #when the condition is meet, change to next state
                        state = 3
                        print('\nNow turn to state 3')
                    
                #take the image information of the target
                x, y, w, h = cv2.boundingRect(contours)
                #draw a rectangular to show the position of detected target
                cv2.rectangle(images,(x, y), (x+w, y+h),(0,0,255),1)
                #Detect the centroid of the target
                moments = cv2.moments(contours)
                #extract the center value of target
                if moments['m00'] > 0.0:
                    centerX = int(moments['m10']/moments['m00'])
                    centerY = int(moments['m01']/moments['m00'])
                    cv2.circle(images,(centerX, centerY), 2,(0,0,255,2))
                    
                # Following is the remote API client side:
                res,replyData=sim.simxQuery(clientID,'request','send me a 100','reply',50)                
                if res==sim.simx_return_ok:
                    print ("The reply is: %s", replyData)
    
            #Show processed images in OpenCV window
            cv2.imshow('Camera', images)
            
        elif res == sim.simx_return_novalue_flag:
            #Camera has not started or is not returning images
            print("No image yet")
            pass
        else:
            #Something else has happened
            print("Unexpected error returned", res)
        
        #At start the robot should align with the target
        if state == 1:
            print('interval now is ', (centerX - midX))
            #detect the intercal between the centre of camera and the target
            if centerX - midX >=10 :
                #make the robot turn slowly
                joint1Angle -= 0.05
                joint2Angle += 0.05
                print('\nNow in state 1')
            elif centerX - midX <=-10 :
                joint1Angle += 0.05
                joint2Angle -= 0.05
                print('\nNow in state 1')
            elif (centerX - midX<10 and centerX - midX>0) or (centerX - midX<0 and centerX - midX>-10):
                #when the condition is met, change to next state
                state = 2
                print('\nNow turn to state 2')
        #this state realign with target in a near distiance
        #the principle is the same with state 1
        if state == 3:
            print('interval is ', (centerX - midX))
            if centerX - midX >=5 :
                joint1Angle -= 0.05
                joint2Angle += 0.05
                print('\nNow in state 3')
            elif centerX - midX <=-5 :
                joint1Angle += 0.05
                joint2Angle -= 0.05
                print('\nNow in state 3')
            elif (centerX - midX<5 and centerX - midX>0) or (centerX - midX<0 and centerX - midX>-5):
                state = 4
                print('\nNow turn to state 4')
            
        #Read keypresses for external control 
        keypress = cv2.waitKey(1) & 0xFF #will read a value of 255 if no key is pressed
        
        #when all conditions are met, start to grib Mr.York
        if state == 4:
            print('\nNow in state 4')
            #Update joints value with changed arrange
            joint1Angle += StepMatrix[counter][0]
            joint2Angle += StepMatrix[counter][1]
            joint3Angle += StepMatrix[counter][2]
            joint4Angle += StepMatrix[counter][3]
            joint5Angle += StepMatrix[counter][4]
            finger1Move += StepMatrix[counter][5]
            finger2Move += StepMatrix[counter][6]
            finger3Move += StepMatrix[counter][7]
            finger4Move += StepMatrix[counter][8]
            #change counter to go to next step
            counter += 1
            
        #Set actuators on mobile robot
        sim.simxSetJointTargetPosition(clientID, joint1, joint1Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint2, joint2Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint3, joint3Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint4, joint4Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint5, joint5Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, finger1, finger1Move, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, finger2, finger2Move, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, finger3, finger3Move, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, finger4, finger4Move, sim.simx_opmode_oneshot)

    #End simulation
    sim.simxFinish(clientID)

else:
    print('Could not connect to remote API server')

#Close all simulation elements
sim.simxFinish(clientID)
cv2.destroyAllWindows()
print('Simulation ended')
