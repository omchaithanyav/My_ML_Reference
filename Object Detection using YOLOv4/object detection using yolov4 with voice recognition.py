import os
import speech_recognition as sr 
import playsound 
from gtts import gTTS 
import cv2


num = 1
def voice(output):
    global num
  
    num += 1
    print("VoiceBot: ", output)
  
    toSpeak = gTTS(text = output, lang ='en', slow = False)
    
    file = str(num)+".mp3" 
    toSpeak.save(file)
      
    playsound.playsound(file, True) 
    os.remove(file)
    
    
    
def getting_audio():
  
    rObject = sr.Recognizer()
    audio = ''
  
    with sr.Microphone() as source:
        print("Say something")
          
        audio = rObject.listen(source, phrase_time_limit = 3) 
  
    try:
        text = rObject.recognize_google(audio, language ='en-US')
        print("user: ", text)
        return text
  
    except:
  
        voice("Sorry, I Could not understand ,PLease do try again!")
        return getting_audio()


def video_obj_detect():
    
    video = 'video.mp4'
    
    Conf_threshold = 0.7
    NMS_threshold = 0.4
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    class_name = []
    with open('coco.names', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg') 
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size = (416,416), scale = 1/255, swapRB = True)
    
    cap = cv2.VideoCapture(video)
    
    while(True):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)] 
            label = "%s : %f" % (class_name[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)
            
        cv2.imshow('frame',frame) 
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            voice("say exit to stop the object detection")
            text = getting_audio()
            
            if 'exit' in str(text):
                break
            
        
    cap.release()
    cv2.destroyAllWindows() 
    
   
def image_obj_detect():
    import cv2
    image = 'img121.jpg'
    
    img = cv2.imread(image)
   
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()
    
    net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
   
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
   
    for (classId, score, box) in zip(classIds, scores, boxes):
       
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)
       
        text = '%s: %.2f' % (classes[classId[0]], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)
    
    cv2.imshow('Image', img)  
   
    key = cv2.waitKey(0)
    
    cv2.destroyAllWindows()
        
    
    

class speak():
    
    if __name__ == "__main__":
        
        
        voice("Hello")
        

        while(True):
          
                voice("just say, 'detect' to start object detection, or say exit to terminate the program")
                text = getting_audio()
                
                if text == 0:
                        break
                    
                elif "detect" in str(text) or "detection" in str(text) or "detecting" in str(text):
                        voice("you want object detection in video or in image")
                        text = getting_audio()
                        if "video" in str(text):
                            video_obj_detect()
                        elif "image" in str(text):
                            image_obj_detect()
                        else:
                            voice("sorry, can't help you with that")
                    
      
                elif "exit" in str(text) or "bye" in str(text) or "stop" in str(text) or "terminate" in str(text):
                        voice("Bye bye")
                        break
                    
                else:
                    voice('Sorry cant help you with that.')
                    break   
                    
               
              
        
            
               
       
    
    


