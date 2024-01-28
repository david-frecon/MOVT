from KalmanFilter import KalmanFilter
import numpy as np
from Detector import detect
import cv2

kFilter = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
video = cv2.VideoCapture('randomball.avi')
last_center = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    centers = detect(frame)
    for center in centers:
        kFilter.predict()
        kFilter.update(center)
        
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
        predicted = (int(kFilter.xk_hat[0][0]), int(kFilter.xk_hat[1][0]))
        cv2.rectangle(frame, (int(predicted[0])-15, int(predicted[1])-15), (int(predicted[0])+15, int(predicted[1])+15), (0, 0, 255), 2)
        xk, yk = kFilter.xk[0][0], kFilter.xk[1][0]
        cv2.rectangle(frame, (int(xk)-15, int(yk)-15), (int(xk)+15, int(yk)+15), (255, 255, 0), 2)
        last_center.append(predicted)
        if len(last_center) >  2:
            for i in range(1, len(last_center) - 1):
                cv2.line(frame, (int(last_center[i][0]), int(last_center[i][1])), (int(last_center[i+1][0]), int(last_center[i+1][1])), (0, 0, 255), 2)
    
    cv2.imshow('frame', frame)
    cv2.waitKey(50)

