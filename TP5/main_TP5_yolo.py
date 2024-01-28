import pandas as pd
import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter
from torchvision import transforms, models, datasets
from torchsummary import summary
from ultralytics import YOLO
from PIL import Image

def IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1min, y1min, x1max, y1max = x1, y1, x1 + w1, y1 + h1
    x2min, y2min, x2max, y2max = x2, y2, x2 + w2, y2 + h2

    xmin = max(x1min, x2min)
    xmax = min(x1max, x2max)
    ymin = max(y1min, y2min)
    ymax = min(y1max, y2max)

    intersection = max(0, (xmax - xmin) * (ymax - ymin))
    union = w1 * h1 + w2 * h2 - intersection
    return np.abs(intersection / union)


def similarity(boxes_prev, boxes_nxt, model):
    sim = np.zeros((len(boxes_prev), len(boxes_nxt)))
    for i in range(len(boxes_prev)):
        for j in range(len(boxes_nxt)):
            sim[i][j] = IoU(boxes_prev[i], boxes_nxt[j])
    return sim


def nb_obj(frame, sigma=40):
    nb = 0
    for i in range(len(frame)):
        if (frame.iloc[i].iloc[6] < sigma):
            continue
        nb += 1
    return nb

def yolo_to_boxes(model, label):
    img = Image.open("ADL-Rundle-6/img1/" + str(int(label)).zfill(6) + ".jpg")
    results = model.predict(source=img)
    boxes = []
    for res in range(len(results)):
        for box in results[res].boxes:
            if box.cls == 0 and box.conf > 0.4:
                boxes.append(box.xywh[0].detach().cpu().numpy())
    return boxes

def frame_to_boxes(frame, sigma=40):
    nb = nb_obj(frame, sigma)
    boxes = np.zeros((nb, 4))
    for i in range(len(frame)):
        if (frame.iloc[i].iloc[6] < sigma):
            continue
        boxes[i][0] = frame.iloc[i].iloc[2]
        boxes[i][1] = frame.iloc[i].iloc[3]
        boxes[i][2] = frame.iloc[i].iloc[4]
        boxes[i][3] = frame.iloc[i].iloc[5]
    return boxes


def greedy_id(sim_matrix, id_prev, sig_iou=0.4):
    id_next = []
    lines_used = []
    for i in range(sim_matrix.shape[1]):
        if len(id_next) == sim_matrix.shape[0]:
            break
        max_col = np.argsort(-(sim_matrix[:, i]))
        for col in max_col:
            if sim_matrix[col][i] < sig_iou:
                break
            if col not in lines_used:
                id_next.append((i, id_prev[col]))
                lines_used.append(col)
                break
    return id_next


def hungarian_id(sim_matrix, id_prev, sig_iou=0.4):
    ## use linear_sum_assignment from scipy.optimize
    clear_sim_matrix = sim_matrix.copy()
    for i in range(len(clear_sim_matrix)):
        for j in range(len(clear_sim_matrix[i])):
            if clear_sim_matrix[i][j] < sig_iou:
                clear_sim_matrix[i][j] = 0
    rows, cols = linear_sum_assignment(-clear_sim_matrix)
    couples = []
    for i in range(len(rows)):
        if clear_sim_matrix[rows[i]][cols[i]] != 0:
            couples.append((cols[i], id_prev[rows[i]]))
    return couples


def update_tracks(id_max, couples, nb_obj):
    new_tracks = []
    for i in range(nb_obj):
        not_find = True
        for col, id in couples:
            if i == col:
                new_tracks.append(id)
                not_find = False
        if not_find:
            id_max += 1
            new_tracks.append(id_max)
    return new_tracks


def display(boxes, labels, tacks, frame, kalmanboxes, kalman):
    image_name = "ADL-Rundle-6/img1/" + str(int(labels)).zfill(6) + ".jpg"
    img = cv2.imread(image_name)
    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][0]) + int(boxes[i][2]), int(boxes[i][1]) + int(boxes[i][3])), (0, 255, 0), 2)
        xe, ye = kalman[tacks[i]].xk[0][0], kalman[tacks[i]].xk[1][0]
        cv2.rectangle(img, (int(xe - kalmanboxes[tacks[i]][2] / 2), int(ye - kalmanboxes[tacks[i]][3] / 2)),
                      (int(xe + kalmanboxes[tacks[i]][2] / 2), int(ye + kalmanboxes[tacks[i]][3] / 2)), (0, 0, 255), 2)
        cv2.putText(img, str(tacks[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                    2, cv2.LINE_AA)
        #cv2.putText(img, str(int(frame.iloc[i].iloc[6])), (int(boxes[i][0]) + int(boxes[i][2]), int(boxes[i][1])),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('jsp', img)

def generate_model_resnet():
    model = models.resnet18(pretrained=True)
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    summary(model, (3, 224, 224))
    return model

def generate_model_mobilenet2():
    model = models.mobilenet_v2(pretrained=True)
    # Supprimer la dernière couche entièrement connectée (classifier)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Gel des paramètres du modèle pré-entraîné
    for p in model.parameters():
        p.requires_grad = False
    summary(model, input_size=(3, 224, 224))
    return model


def generate_model_mobilenet():
    model = models.mobilenet_v3_small(pretrained=True)
    # Remove the classifier layer
    model.classifier = torch.nn.Identity()
    # Freeze the parameters of the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    return model



def compute_features(model, label, boxes):
    features = []
    image_name = "ADL-Rundle-6/img1/" + str(int(label)).zfill(6) + ".jpg"
    image = cv2.imread(image_name)
    for box in boxes:
        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        crop = image[int(y):int(y + h), int(x):int(x + w)]
        crop = cv2.resize(crop, (224, 224))
        crop = transforms.ToTensor()(crop)
        crop = torch.unsqueeze(crop, 0)
        feature = model(crop)
        feature = torch.squeeze(feature)
        feature = feature.detach().numpy()
        features.append(feature)
    return np.array(features)  # Convertir la liste de caractéristiques en une matrice

def compute_sim_features(model, label_prev, label, boxes_prev, boxes):
    features_prev = compute_features(model, label_prev, boxes_prev)
    features = compute_features(model, label, boxes)
    sim = np.zeros((len(features_prev), len(features)))
    for i, feature_prev in enumerate(features_prev):
        for j, feature in enumerate(features):
            # cosine similarity
            sim[i][j] = np.dot(feature_prev, feature) / (np.linalg.norm(feature_prev) * np.linalg.norm(feature))
    return sim


def sim_fusion(sim1, sim2, alpha=0.5):
    return alpha * sim1 + (1 - alpha) * sim2

def find_center(box):
    x, y, w, h = box
    return [[x + w / 2], [y + h / 2]]


def save_detection(track, kalman, kalmanboxes, boxes):
    for i in range(len(track)):
        if track[i] not in kalman.keys():
            kalman[track[i]] = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
            kalmanboxes[track[i]] = boxes[i]
            kalman[track[i]].predict()
            kalman[track[i]].update(find_center(boxes[i]))
        else:
            kalman[track[i]].predict()
            kalman[track[i]].update(find_center(boxes[i]))
            kalmanboxes[track[i]] = boxes[i]


def kalman_to_boxes(kalman, kalmanboxes, track):
    boxes = []
    for i in range(len(track)):
        w, h = kalmanboxes[track[i]][2], kalmanboxes[track[i]][3]
        xe, ye = kalman[track[i]].xk[0][0], kalman[track[i]].xk[1][0]
        boxes.append([xe - w / 2, ye - h / 2, w, h])
    return boxes


def main():
    kalman = {}
    kalmanboxes = {}
    model = generate_model_mobilenet()
    model = YOLO("yolov8m.pt")
    sigma = 15
    det = pd.read_csv('ADL-Rundle-6/det/det.txt')
    det_grouped_frame = det.groupby('frame')
    nb_group = det_grouped_frame.ngroups
    # init tracking
    frame1 = det_grouped_frame.get_group(1)
    #boxes2 = frame_to_boxes(frame1, sigma)
    boxes2 = yolo_to_boxes(model, frame1.iloc[0].iloc[0])
    tracks = update_tracks(0, [], len(boxes2))
    save_detection(tracks, kalman, kalmanboxes, boxes2)  # TP 4
    last_frame = frame1
    for i in range(2, nb_group, 1):
        ## display
        display(boxes2, last_frame.iloc[0].iloc[0], tracks, last_frame, kalmanboxes, kalman)
        k = cv2.waitKey(0)
        if k == 27:
            break
        act_frame = det_grouped_frame.get_group(i)
        # boxes1 = frame_to_boxes(last_frame,sigma)
        boxes1 = kalman_to_boxes(kalman, kalmanboxes, tracks)
        #boxes2 = frame_to_boxes(act_frame, sigma)
        boxes2 = yolo_to_boxes(model, act_frame.iloc[0].iloc[0])
        labels = act_frame.iloc[0].iloc[0]
        features_matrix = compute_sim_features(model, last_frame.iloc[0].iloc[0], labels, boxes1, boxes2)
        sim_matrix = similarity(boxes1, boxes2, model)
        sim_matrix = sim_fusion(sim_matrix, features_matrix)
        couples = hungarian_id(sim_matrix, tracks)
        tracks = update_tracks(max(tracks), couples, len(boxes2))
        save_detection(tracks, kalman, kalmanboxes, boxes2)
        last_frame = act_frame


main()