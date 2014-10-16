'''
Created on Oct 14, 2014

@author: Dereck
'''
import cv2

if __name__ == '__main__':
    print('Hello World')
    
    cam = cv2.VideoCapture()
    cam.open("http://10.25.86.11/mjpg/video.mjpg")
    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)
        
        
        
                