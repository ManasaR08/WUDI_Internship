import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

json_file = open('fer.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("fer.h5")

gender_model = load_model('pre-trained/gender_detection.model')

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

WIDTH = 48
HEIGHT = 48
x=None
y=None
labels_emo = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
labels_gen = ['Man','Woman']

cv2.namedWindow('cam')
cap = cv2.VideoCapture(0)
while True:

    ret, img_rgb = cap.read()
    img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        roi_gray = img_gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= loaded_model.predict(cropped_img)
        cv2.putText(img_rgb, labels_emo[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels_emo[int(np.argmax(yhat))])


    for (x, y, w, h) in faces:
        roi_rgb = img_rgb[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_rgb, (96,96))
        cropped_img = cropped_img.astype("float") / 255.0
        cropped_img = img_to_array(cropped_img)
        cropped_img = np.expand_dims(cropped_img, axis=0)
        #predicting the gender
        ghat= gender_model.predict(cropped_img)
        cv2.putText(img_rgb, labels_gen[int(np.argmax(ghat))], (x, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("Gender: "+labels_gen[int(np.argmax(ghat))])


    cv2.imshow('Emotion and Gender', img_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
