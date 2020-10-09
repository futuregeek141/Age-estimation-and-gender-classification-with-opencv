import cv2
import math
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,help="path to input image")
#parser.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"	
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

image = cv2.imread("D:/Files/Sem 7/SGP/basic/woman1.jpg")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

faceNet.setInput(blob)
detections = faceNet.forward()

box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
(startX, startY, endX, endY) = box.astype("int")
		
face = image[startY:endY, startX:endX]
faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)    

genderNet.setInput(blob)
genderPreds=genderNet.forward()
gender=genderList[genderPreds[0].argmax()]
print(f'Gender: {gender}')
        
y = startY - 10 if startY - 10 > 10 else startY + 10
		
cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
cv2.putText(image, gender, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
cv2.imshow("Image", image)
cv2.waitKey(0)