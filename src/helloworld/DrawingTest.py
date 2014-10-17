'''
Created on Oct 14, 2014

@author: Dereck Wonnacott
'''
import cv2
from cv2 import imshow
import numpy as np

hsv   = []

# mouse callback function
def Pixel_Picker(event,x,y,flags,param):
    global target
    if event == cv2.EVENT_FLAG_LBUTTON:
        pixel = hsv[y, x]
        target = pixel

def passfun(x):
    pass

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
    
    # Mouse callback
    cv2.namedWindow('Source video')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Result')
    cv2.setMouseCallback('Source video', Pixel_Picker)
    cv2.setMouseCallback('Mask', Pixel_Picker)
    cv2.setMouseCallback('Result', Pixel_Picker)
    
    # Trackbar
    cv2.createTrackbar('Mask Width1','Source video',0,255,passfun)
    cv2.createTrackbar('Mask Width2','Source video',0,255,passfun)
    cv2.createTrackbar('Mask Width3','Source video',0,255,passfun)
    cv2.setTrackbarPos('Mask Width1','Source video', 6)
    cv2.setTrackbarPos('Mask Width2','Source video', 255)
    cv2.setTrackbarPos('Mask Width3','Source video', 120)

    global target  
    target = np.array([100, 142, 208])
    
    # Grab frames
    while(True):
        ret, frame = cam.read()            
        imshow("Source video", frame)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
         
        # define range of blue color in HSV
        print(">>>", target)
        w = cv2.getTrackbarPos('Mask Width1','Source video')
        x = cv2.getTrackbarPos('Mask Width2','Source video')
        y = cv2.getTrackbarPos('Mask Width3','Source video')
        lower_blue = np.array([target[0]-w,target[1]-x,target[2]-y])
        upper_blue = np.array([target[0]+w,target[1]+x,target[2]+y])

        
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)      
        imshow("Mask", mask)
        
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)   
        imshow("Result", res)
        
         
        # Line Demo       
        #cv2.line(frame, (390, 300), (410, 300), (0,0,255), 3)  
        #cv2.line(frame, (400, 290), (400, 310), (0,0,255), 3)              
        #imshow("Edited video", frame) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
      
        #time.sleep(0.2)
           
# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()