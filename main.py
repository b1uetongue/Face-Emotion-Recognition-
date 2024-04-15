import cv2
import numpy as np
from keras.models import model_from_json
#plt.rcParams['font.sans-serif'] = ['SimHei'] 
with open('fer.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights('fer-200.h5')
face_prototxt = 'deploy.prototxt.txt'
face_model = 'res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
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
def process_image_file(file_path):
    img = cv2.imread(file_path)
    processed_img = process_image(img)
    cv2.namedWindow('image', 0)
    cv2.imshow('image', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def process_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_image(frame)
        resize_frame = cv2.resize(processed_frame, (1000, 700))
        cv2.imshow('frame', resize_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
def process_video(video_path, output_path):  # 添加 output_path 参数
    cap = cv2.VideoCapture(video_path)
    # 获取视频的帧率、大小等信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # 定义VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者使用其他的视频编码格式
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_image(frame)
        resize_frame = cv2.resize(processed_frame, frame_size)
        cv2.imshow('video', resize_frame)
        
        # 将帧写入VideoWriter
        out.write(resize_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    out.release()  # 释放VideoWriter资源
    cv2.destroyAllWindows()
def main():
    while True:
        mode = input("选择模式（图像/相机/视频，退出请输入'q'）：")
        if mode == "q":
            break
        elif mode == "图像":
            file_path = input("请输入图片的路径：")
            process_image_file(file_path)
        elif mode == "相机":
            process_camera()
        elif mode == "视频":
            video_path = input("请输入视频的路径：")
            output_path = input("请输入保存视频的路径（包括文件名和扩展名，例如 output.avi）：")
            process_video(video_path, output_path)
        else:
            print("模式无效。使用‘图像’、‘相机’或‘视频’。")

if __name__ == "__main__":
    main()

