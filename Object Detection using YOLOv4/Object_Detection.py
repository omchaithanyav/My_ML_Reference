import cv2 

Conf_threshold = 0.7  # Confidence score threshold (in our case 0.7) used to filter boxes by confidence scores
NMS_threshold = 0.4   # The most appropriate boxes are selected by using non-maximum suppression (NMS). NMS is controlled by threshold (0.4 in our case).
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)] # (0, 255, 0) = green, (0, 0, 255) = blue, (255, 0, 0) = red

class_name = []
with open('coco.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()] # read the class names
    #print(class_name)
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg') # read network configuration and pre-trained weights
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net) # initialize detection model  
model.setInputParams(size = (416,416), scale = 1/255, swapRB = True) # set input parameters
# 1/255 scale factor defines that pixel values will be scaled from 0 to 1.
# Given image will be resized to the size of 416×416 without cropping.
# The swapRB parameter defines that first and last channels will be swapped because OpenCV uses BGR.


cap = cv2.VideoCapture(0)  # get a video capture object for the camera or for a video file.

while(True):
    ret, frame = cap.read() 
    # ret is a boolean variable that returns true if the frame is available. 
    # frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
    # cap.read() returns a bool (True/False). If frame is read correctly, it will be True. 
    if ret == False:
        break
   
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold) # detecting objects using the model... and from this we get the classes arrays, scores arrays, arrays of boxes coordinates
    
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)] # assigns unique color to each box.
        #print(int(classid))
        label = class_name[classid[0]] # label (label of class name and score) for each object.
        cv2.rectangle(frame, box, color, 2) # cv2.rectangle(image, coordinates of the rectangle box, color, thickness of box)
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1) #  cv2.putText(image, text, org(coordinates), font, font_Scale, color, font_thickness)
        
    cv2.imshow('Object Detection',frame)
    # Use cv2.imshow() method to show the frames in the video.
    # frame is a window with name 'frame'
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        break
    # Breaks the loop when the user clicks a specific key.

cap.release() # Closes video file or capturing device
cv2.destroyAllWindows() 
# If you have multiple windows open and you do not need those to be open, you can use cv2.destroyAllWindows() to close those all. 
# using 'cv2.destroyAllWindows()' function, is a good practice.
 

