import cv2
import numpy as np

# Create an object to read camera video 
cap = cv2.VideoCapture(0)

video_cod = cv2.VideoWriter_fourcc(*'XVID')
video_output= cv2.VideoWriter('captured_video.avi',
                      video_cod,
                      10,
                      (640,480))

while(True):
    _, frame = cap.read()
                
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define blue color range
    light_blue = np.array([110,50,50])
    dark_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, light_blue, dark_blue)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(frame,frame, mask= mask)
    
    # Write the frame into the file 'captured_video.avi'
    video_output.write(output)

    # Display the frame, saved in the file   
    cv2.imshow('output',output)

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('Q'):
      break

# release video capture
# and video write objects
cap.release()
video_output.release()

# Closes all the frames
cv2.destroyAllWindows() 

print("The video was successfully saved")   
