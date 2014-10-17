'''
Created on Oct 14, 2014

@author: Dereck Wonnacott
'''
import cv2
from cv2 import imshow

if __name__ == '__main__':
    print('Hello World')
    
    # Connect to video Source
    cam = cv2.VideoCapture()
    cam.open("c:\users\dereck\desktop\capture\output.avi")
    #cam.open("http://10.25.86.11/mjpg/video.mjpg")
    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)
    
    # Grab frames
    while(True):
        ret, frame = cam.read()            
        imshow("Source video", frame)                
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
            
# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()