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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masked = cv2.bitwise_and(frame2, frame2, mask=mask)
               
        # Contours?
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        
        # filter out small blobs
        contours2 = []
        for c in contours:
            
            p = cv2.arcLength(c, True)
            a = cv2.contourArea(c)
            
            if a < 200: 
                continue
                
            rect = cv2.minAreaRect(c)                      
            w = rect[1][0]
            h = rect[1][1]   
            
            if abs((h/w) - 1) > 0.2:
                continue
            
            contours2.append(c)
                
        
        # Show feature data for each contour within threshold
        con = frame2.copy()
        cv2.drawContours(con, contours, -1, (0,0,255), 1)   
        cv2.drawContours(con, contours2, -1, (0,255,0), 1)   
     
        for c in contours2:    
            # Bounding box
            rect = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(con, [box], 0, (255,0,0), 2)
            
            w = rect[1][0]
            h = rect[1][1]     
                        
            # Area and perimeter
            p = cv2.arcLength(c, True)
            a = cv2.contourArea(c)
                               
            # Display the features for each blob     
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            disp = "P: %d, A: %d, W:, %d H: %d" % (p, a, w, h)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(con, disp, (cx,cy), font, 0.4, (255,255,255), 1)
            
        
        
        
        # Show the results
        #imshow("Source video", frame)  
        #imshow("Calibrated Video", frame2)
        #imshow("Masked", masked)
             
        imshow("Contours", con)
        
        
        # Exit Nicely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
      
        #time.sleep(0.2)
           
# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()