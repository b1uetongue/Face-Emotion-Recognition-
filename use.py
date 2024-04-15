import cv2
import numpy as np
from keras.models import model_from_json

# 加载面部表情识别模型
with open('fer-1.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights('fer-1.h5')

# 加载人脸检测模型
face_prototxt = 'deploy.prototxt.txt'
face_model = 'res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)

# 定义情感类别和颜色
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotion_colors = {
    'angry': (0, 0, 255),       # 红色
    'disgust': (0, 255, 0),     # 绿色
    'fear': (255, 0, 0),        # 蓝色
    'happy': (255, 255, 0),     # 黄色
    'sad': (255, 0, 255),       # 紫色
    'surprise': (0, 255, 255),  # 青色
    'neutral': (255, 255, 255)  # 白色
}

font = cv2.FONT_HERSHEY_SIMPLEX

# 图像处理函数
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    faces = face_net.forward()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            face = gray[startY:endY, startX:endX]
            face = cv2.resize(face, (48, 48))
            pixels = np.expand_dims(face, axis=0)
            pixels = pixels / 255.0
            pred = model.predict(pixels)
            max_index = np.argmax(pred[0])
            pred_emotion = emotions[max_index]
            color = emotion_colors.get(pred_emotion, (255, 255, 255))
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 3)
            cv2.putText(img, pred_emotion, (int(startX), int(startY)), font, 1, color, 2)
    return img

# 处理单张图像文件
def process_image_file(file_path):
    img = cv2.imread(file_path)
    processed_img = process_image(img)
    cv2.namedWindow('image', 0)
    cv2.imshow('image', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用处理函数
process_image_file('2.jpg')
