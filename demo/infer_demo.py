import cv2
from ultralytics import YOLO
 
model = YOLO(model="runs/segment/train18/weights/best.pt")
model.export(format="tflite", int8=True)


'''
camera_no = 0
cap = cv2.VideoCapture(camera_no)
 
while cap.isOpened():
    res, frame = cap.read()
    if res:
        results = model(frame)

        annotated_frame = results[0].plot(boxes=False, probs= True, labels=True)
 
        cv2.imshow(winname="YOLOV8", mat=annotated_frame)
 
        if cv2.waitKey(1) == 27:
            break
 
    else:
        break
 
cap.release()
cv2.destroyAllWindows() 

'''
'''
img = cv2.imread("t.jpg")
results = model(img)
annotated_frame = results[0].plot(boxes=True, probs= True, labels=True)
cv2.imshow(winname="YOLOV8", mat=annotated_frame)
cv2.waitKey(0)
'''