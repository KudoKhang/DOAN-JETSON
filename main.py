import time
import cv2
import numpy as np
import torch
import os
import sys
import math

# Import Yolov5
os.sys.path
sys.path.insert(0, "yolov5")
from yolov5.models.experimental import attempt_load
from yolov5.yolo_utils import non_max_suppression, scale_coords, resize_image

# Init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("./Sources/weights/detect_orange.pt", map_location=device)

# Hyperparameter
size_convert = 640  
conf_thres = 0.4
iou_thres = 0.4

YELLOW = (0, 255, 255)
BLUE = (255, 225, 0)
ORANGE = (0, 128, 255)
LV1 = (255, 191, 0)
LV2 = (140, 230, 240)
FPS = (50, 170, 50)
LINE = (255, 248, 240)
NUM_L1 = 0
NUM_L2 = 0
THRESH_LINE1 = 400
THRESH_LINE2 = 200
DEGREE = 120

# Init ServoKit
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
kit.servo[0].angle=DEGREE
kit.servo[1].angle=DEGREE

def control_servo(pin):
    for degree in range(DEGREE, 0, -1):
        kit.servo[pin].angle=degree
    s = time.time()
    while time.time() < s + 0.3:
        pass
    for degree in range(0, DEGREE, 1):
        kit.servo[pin].angle=degree

def check_type(img_path):
    if type(img_path) == str:
        if img_path.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(img_path)
        else:
            raise Exception("Please input a image file")
    elif type(img_path) == np.ndarray:
        img = img_path
    return img

def classify(bboxs, thresh=35000):
    area = (bboxs[2] - bboxs[0]) * (bboxs[3] - bboxs[1])
    return 1 if area > thresh else 2

def draw(img, bbox):
    x1, y1, x2, y2 = bbox
    lv = classify(bbox, 35000)

    cv2.putText(img, 'ORANGE_' + str(lv), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, ORANGE, 2)

    cx, cy,  = get_center(bbox)

    a = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 4)
    
    cv2.line(img, (x1, y1), (x1 + a, y1), YELLOW, 2)
    cv2.line(img, (x1, y1), (x1, y1 + a), BLUE, 2)
    
    cv2.line(img, (x2, y1), (x2, y1 + a), YELLOW, 2)
    cv2.line(img, (x2, y1), (x2 - a, y1), BLUE, 2)
    
    cv2.line(img, (x2, y2), (x2 - a, y2), YELLOW, 2)
    cv2.line(img, (x2, y2), (x2, y2 - a), BLUE, 2)

    cv2.line(img, (x1, y2), (x1 + a, y2), BLUE, 2)
    cv2.line(img, (x1, y2), (x1, y2 - a), YELLOW, 2)

    cv2.circle(img, (cx, cy), 2, (255, 255, 153), -1)

def show_info(frame, fps):
    print(fps)
    cv2.putText(frame, "FPS: " + str(fps), (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.75, FPS, 2)
    cv2.putText(frame, f'TOTAL ORANGE 1: {NUM_L1}', (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, LV1, 2)
    cv2.putText(frame, '-------------', (20, 103), cv2.FONT_HERSHEY_COMPLEX, 0.75, LINE, 2)
    cv2.putText(frame, f'TOTAL ORANGE 2: {NUM_L2}', (20, 125), cv2.FONT_HERSHEY_COMPLEX, 0.75, LV2, 2)
    
    h, w, _ = frame.shape
    cv2.line(frame, (100, THRESH_LINE2), (w - 150, THRESH_LINE2), LV2, 2)    
    cv2.line(frame, (100, THRESH_LINE1), (w - 150, THRESH_LINE1), LV1, 2)    

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = int((x2 - x1) / 2) + x1, int((y2 - y1) / 2) + y1
    return cx, cy

def count(bbox):
    cx, cy = get_center(bbox)

def push(bbox):
    cx, cy = get_center(bbox)
    lv = classify(bbox)
    print(lv)
    if cy < THRESH_LINE1 and cy > THRESH_LINE1 - 50:
        if lv == 1:
            print('Push: L1')
            control_servo(pin=0)
            
    if cy < THRESH_LINE2 and cy > THRESH_LINE2 - 50:
        if lv == 2:
            print('Push: L2')
            control_servo(pin=1)

def inference(image):
    image = check_type(image)
    img = resize_image(image.copy(), size_convert).to(device)
    with torch.no_grad():
        pred = model(img[None, :])[0]
        det = non_max_suppression(pred, conf_thres, iou_thres)[0] # x1,y1,x2,y2, score, class
        bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], image.shape[:-1]).round().cpu().numpy())# x1, y1, x2, y2    
    return bboxs

def webcam(mode='./Sources/Video/3.mp4'):
    # 0: Camera or Path Video
    cap = cv2.VideoCapture(mode)
    while True:
        _, frame = cap.read()
        if mode == 0:
            frame = cv2.flip(frame, 1)
        start = time.time()
        bboxs = inference(frame)
        if len(bboxs) > 0:
            for bb in bboxs:
                # Drawing bbox
                draw(frame, bb)

                # Push
                push(bb)
                
                # Delay
                t = time.time()
                while time.time() < t + 2:
                    _, frame = cap.read()

        # FPS
        fps = round(1 / (time.time() - start), 2)

        # Show FPS, Numbers...        
        show_info(frame, fps)

        cv2.imshow('RESULT', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            cv2.imwrite(str(time.time()) + '.jpg', frame)
        if k == ord('q'):
            break

def image():
    root = './Sources/final_data/yolo_orange/test/images/'
    path_name = [name for name in os.listdir(root) if name.endswith('jpg')]
    for name in path_name[:1]:
        img = cv2.imread(root + name)
        bboxs = inference(img)
        if len(bboxs) > 0:
            for bb in bboxs:
                draw(img, bb)
        cv2.imshow('RESULT', img)
        cv2.waitKey(0)


#TODO
"""
    - Count orange
        +) Tại thời điểm gạt -> num +=1 
    - Processing in ROI
""" 

if __name__ == '__main__':
    webcam(0)
    # image()
