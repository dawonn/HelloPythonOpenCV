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
roi_points = [(246, 144), (373, 148), (372, 264), (248, 262)]
bins = np.arange(256).reshape(256,1)

# Histogram plot function
def hist_curve(im, mask=None):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],mask,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y


# ROI mouse callback function
def roi_picker(event,x,y,flags,param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        p = (x, y)
        roi_points.append(p)
        
        if len(roi_points) > 4:
            roi_points = roi_points[1:5]
        
        print(roi_points)

# Histogram mouse callback function
def histogram_picker(event,x,y,flags,param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        p = (x, y)
        print(p)

# Dummy CB 
def passfun(x):
    pass

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
    
    # ROI selection callback
    cv2.namedWindow('Calibrated Video')
    cv2.setMouseCallback('Calibrated Video', roi_picker)
    cv2.namedWindow('histogram')
    cv2.setMouseCallback('histogram', histogram_picker)
    
    # Mask Sliders
    cv2.namedWindow('Mask')
    cv2.createTrackbar('Hmin','Mask',0,255, passfun)
    cv2.createTrackbar('Hmax','Mask',0,255, passfun)
    cv2.createTrackbar('Smin','Mask',0,255, passfun)
    cv2.createTrackbar('Smax','Mask',0,255, passfun)
    cv2.createTrackbar('Vmin','Mask',0,255, passfun)
    cv2.createTrackbar('Vmax','Mask',0,255, passfun)    
    cv2.setTrackbarPos('Hmin','Mask', 99)
    cv2.setTrackbarPos('Hmax','Mask', 117)
    cv2.setTrackbarPos('Smin','Mask', 172)
    cv2.setTrackbarPos('Smax','Mask', 255)
    cv2.setTrackbarPos('Vmin','Mask', 0)
    cv2.setTrackbarPos('Vmax','Mask', 255)
    
    # Grab frames
    while(True):
        ret, frame = cam.read()         
        
        # Remove distortion and crop 
        frame2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        frame2 = frame2[y:y+h, x:x+w]
        frame3 = frame2.copy()
        frame4 = frame2.copy()
        
        # Show the selected ROI (Source)
        pts = np.array(roi_points, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame2,[pts],(len(roi_points) == 4),(0,255,0))
        
        # Show the selected ROI (perspective)
        if len(roi_points) == 4:
            pts1 = np.float32(pts)
            pts2 = np.float32([[0,0],[500,0],[500,500],[0,500]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            frame3 = cv2.warpPerspective(frame3,M,(500,500))
        
            # Generate a histogram for the ROI
            hsv = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)
            histo = hist_curve(hsv)
        
        # Image Mask        
        Hmin = cv2.getTrackbarPos('Hmin','Mask')
        Hmax = cv2.getTrackbarPos('Hmax','Mask')
        Smin = cv2.getTrackbarPos('Smin','Mask')
        Smax = cv2.getTrackbarPos('Smax','Mask')
        Vmin = cv2.getTrackbarPos('Vmin','Mask')
        Vmax = cv2.getTrackbarPos('Vmax','Mask')
        
        lower = np.array([Hmin, Smin, Vmin])
        upper = np.array([Hmax, Smax, Vmax])
        hsv = cv2.cvtColor(frame4, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        masked = cv2.bitwise_and(frame4, frame4, mask= mask)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masked2 = cv2.bitwise_and(frame4, frame4, mask= opening)
               
        # Generate a histogram for the masked image
        hsv = cv2.cvtColor(masked2, cv2.COLOR_BGR2HSV)
        histo_masked = hist_curve(hsv, mask)
        
        # Show the results
        imshow("Source video", frame)  
        imshow("Calibrated Video", frame2)
        
        if len(roi_points) == 4:
            imshow("Perspective View", frame3) 
            imshow('histogram', histo)
            
        imshow("Masked", masked)
        imshow("Masked2", masked2)
        imshow("Histogram Masked", histo_masked)
        
        # Exit Nicely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
      
        #time.sleep(0.2)
           
# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()