'''
Created on Oct 17, 2014

@author: Dereck Wonnacott
'''
import cv2
from cv2 import imshow

import numpy as np
import pickle

if __name__ == '__main__':
    print('Hello World')
    
    # Connect to video Source
    cam = cv2.VideoCapture()
    #cam.open("c:\users\dereck\desktop\capture\output.avi")
    cam.open("http://10.25.86.11/mjpg/video.mjpg")
    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)
    
    BoardSize = (9,6)
        
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((BoardSize[0]*BoardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:BoardSize[0],0:BoardSize[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    i = 0
    
    # Grab distorted frames
    while(True):
        ret, frame = cam.read()                      
        Grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        
        ret, corners = cv2.findChessboardCorners(Grayframe, BoardSize)
        
        if ret:                       
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
            cv2.cornerSubPix(Grayframe,corners,(5,5),(-1,-1),criteria)            
            cv2.drawChessboardCorners(frame, BoardSize, corners,ret)
            
            i = i+1;
            if (i % 40 == 0) :
                imgpoints.append(corners)
                objpoints.append(objp)
                print("Added Frame.")
             
        imshow("Source video", frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
    
    
    # Calibrate Optics
    print("Generating Calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, Grayframe.shape[::-1])
    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    
    # saving to file
    CalibFile = "c:\users\dereck\desktop\capture\calibration.npy"
    with open(CalibFile, 'w') as f:
        pickle.dump([mtx, dist, newcameramtx, roi], f)
    
    # Test loading from file
    del mtx, dist, newcameramtx, roi
    
    # loading from file
    with open(CalibFile) as f:
        mtx, dist, newcameramtx, roi = pickle.load(f)
    
    
    # Remove distortion from frames
    while(True):
        ret, frame = cam.read()                      
          
        # Remove Distortion
        newframe = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # crop the image
        x,y,w,h = roi
        newframe = newframe[y:y+h, x:x+w]
        
        # Show the results
        imshow("Source video", frame)  
        imshow("Calibrated video", newframe)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
    
# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()