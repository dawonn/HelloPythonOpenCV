'''
Created on Oct 14, 2014

@author: Dereck Wonnacott

You must start at the top left corner and select each corner in a clockwise rotation
'''
import cv2
from cv2 import imshow
import numpy as np
import pickle

hsv   = []

if __name__ == '__main__':

    # Connect to video Source
    cam = cv2.VideoCapture()
    #cam.open("c:\users\dereck\desktop\capture\output.avi")
    cam.open("http://10.25.86.11/mjpg/video.mjpg")
    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)
    
    # Load Camera Calibration
    CalibFile = "c:\users\dereck\desktop\capture\calibration.npy"
    with open(CalibFile) as f:
        mtx, dist, newcameramtx, roi = pickle.load(f)
    
   

    # Grab frames
    while(True):
        ret, frame = cam.read()         
        
        # Remove distortion and crop 
        frame2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        frame2 = frame2[y:y+h, x:x+w]
                
        # Image Mask        
        Hmin = 99
        Hmax = 117
        Smin = 172
        Smax = 255
        Vmin = 0
        Vmax = 255
        
        lower = np.array([Hmin, Smin, Vmin])
        upper = np.array([Hmax, Smax, Vmax])
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masked = cv2.bitwise_and(frame2, frame2, mask= opening)
               

        # Show the results
        imshow("Source video", frame)  
        imshow("Calibrated Video", frame2)
        imshow("Masked", masked)

        
        # Exit Nicely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
      
        #time.sleep(0.2)
           
# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()