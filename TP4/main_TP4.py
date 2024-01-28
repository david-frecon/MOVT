import pandas as pd
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

def IoU(box1, box2):
    """
        Compute the IoU between two boxes.
        The boxes are expected to be in the format [x, y, w, h].
    """
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

def similarity(boxes_prev, boxes_nxt):
    """
        Compute the similarity matrix between two frames.
        So the similarity between two boxes is the IoU between them.
    """
    sim = np.zeros((len(boxes_prev), len(boxes_nxt)))
    for i in range(len(boxes_prev)):
        for j in range(len(boxes_nxt)):
            sim[i][j] = IoU(boxes_prev[i], boxes_nxt[j])
    return sim 

def nb_obj(frame, sigma = 40):
    nb = 0
    for i in range(len(frame)):
        if (frame.iloc[i].iloc[6] < sigma):
            continue
        nb += 1
    return nb


def frame_to_boxes(frame, sigma = 40):
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

def hungarian_id(sim_matrix, id_prev, sig_iou = 0.4):
    """
    Compute the hungarian algorithm to find the best id for each box.
    """

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
    """
        Update the tracks with the new id. The couples are made from the greedy algorithm.
        So it upadate the tracks with the new id and create new id for the new objects.
    """
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

def display(boxes, labels, tacks, frame, kalmanboxes, kalman, df_gt):
    """
        Display the boxes on the image.
    """
    image_name = "ADL-Rundle-6/img1/" + str(int(labels)).zfill(6) + ".jpg"
    img = cv2.imread(image_name)
    for i in range(len(boxes)):
        df_gt = df_gt._append({'frame': int(labels), 'id': int(tacks[i]), 'x': int(boxes[i][0]), 'y': int(boxes[i][1]),
                               'w': int(boxes[i][2]), 'h': int(boxes[i][3]), 'score': 1, 'class': 1, 'visibility': 1,
                               'unused': 1}, ignore_index=True)
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][0]) + int(boxes[i][2]), int(boxes[i][1]) + int(boxes[i][3])), (0, 255, 0), 2)
        xe, ye = kalman[tacks[i]].xk[0][0], kalman[tacks[i]].xk[1][0]
        cv2.rectangle(img, (int(xe - kalmanboxes[tacks[i]][2]/2), int(ye - kalmanboxes[tacks[i]][3]/2)), (int(xe + kalmanboxes[tacks[i]][2]/2), int(ye + kalmanboxes[tacks[i]][3]/2)), (0, 0, 255), 2)
        cv2.putText(img,  str(tacks[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2, cv2.LINE_AA)
        cv2.putText(img, str(int(frame.iloc[i].iloc[6])), (int(boxes[i][0]) + int(boxes[i][2]) , int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,0), 2, cv2.LINE_AA)
    cv2.imshow('jsp',img)
    return df_gt

def find_center(box):
    """
    Find the center of the box. This is used for the kalman filter.
    """
    x, y, w, h = box
    return [[x + w/2], [y + h/2]]



def save_detection(track, kalman, kalmanboxes, boxes):
    """
    Save the detection in the kalman filter.
    kalman is a dictionnary with the id of the track as key and the kalman filter as value.
    So if we find a new track, we create a new kalman filter, update and predict it.
    If we find an old track, we update and predict it.
    And in every case, we save the box in kalmanboxes.
    """
    for i in range(len(track)):
        if track[i] not in kalman.keys():
            kalman[track[i]] = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
            print(track, len(track))
            print(boxes, len(boxes))
            kalmanboxes[track[i]] = boxes[i]
            kalman[track[i]].predict()
            kalman[track[i]].update(find_center(boxes[i]))
            
        else:
            kalman[track[i]].predict()
            kalman[track[i]].update(find_center(boxes[i]))
            
            

def kalman_to_boxes(kalman, kalmanboxes ,track):
    """
    Generate the boxes from the kalman filter.
    """
    boxes = []
    for i in range(len(track)):
        w, h = kalmanboxes[track[i]][2], kalmanboxes[track[i]][3]
        xe, ye = kalman[track[i]].xk[0][0], kalman[track[i]].xk[1][0]
        boxes.append([xe - w/2, ye - h/2, w, h])
    return boxes

def main():
    kalman = {}
    kalmanboxes = {}
    sigma = 15
    det = pd.read_csv('ADL-Rundle-6/det/det.txt')
    df_gt = pd.DataFrame(columns=['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'class', 'visibility', 'unused'])
    det_grouped_frame = det.groupby('frame')
    nb_group = det_grouped_frame.ngroups
    # init tracking
    frame1 = det_grouped_frame.get_group(1)
    boxes2 = frame_to_boxes(frame1,sigma)
    tracks = update_tracks(0, [], len(boxes2))
    save_detection(tracks, kalman, kalmanboxes, boxes2) # TP 4
    last_frame = frame1
    for i in range(2,nb_group,1):
        ## display 
        df_gt = display(boxes2, last_frame.iloc[0].iloc[0], tracks, last_frame,  kalmanboxes, kalman, df_gt)
        k = cv2.waitKey(0)
        if k == 27:
            break
        act_frame = det_grouped_frame.get_group(i)
        boxes1 = kalman_to_boxes(kalman, kalmanboxes, tracks)
        boxes2 = frame_to_boxes(act_frame,sigma)
        sim_matrix = similarity(boxes1, boxes2)
        couples = hungarian_id(sim_matrix, tracks)
        tracks = update_tracks(max(tracks), couples, len(boxes2))
        save_detection(tracks, kalman, kalmanboxes, boxes2)
        last_frame = act_frame
    df_gt.to_csv('ADL-Rundle-6/det/det_gt.txt', index=False)

main()