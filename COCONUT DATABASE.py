import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))

model.setInputSize(320,320)
model.setInputScale(1.0/127.5) #255/2 = 127.5
model.setInputMean((127.5,127.5,127.5)) #mobile net
model.setInputSwapRB(True)

# import matplotlib.image as mpimg
# img = mpimg.imread('img1.png')
# imgplot = plt.imshow(img)
# plt.show()

# imgplot = plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.show()

img = cv2.imread('img1.png')
#plt.imshow(img)  #BGR Photo
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  #RGB Original Photo colour


ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.5) #conf threshold = accuracy

print(ClassIndex)

font_scale = 0.00001
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf ,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale ,color=(0,255,0),thickness=1)


imgplot = plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
plt.show()


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Video")


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame =cap.read()
    ClassIndex, confidence, bbox = model.detect(frame,confThreshold=0.55) 
    print(ClassIndex)
    
    if (len(ClassIndex)!=0):
        for ClassInd, conf ,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale ,color=(0,255,0),thickness=1)
    cv2.imshow('Object Dection Tutorial',frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()