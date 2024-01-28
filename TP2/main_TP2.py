import pandas as pd
import numpy as np
import cv2

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
    """
    Convert a frame that is a pandas dataframe to a numpy array of boxes.
    """
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

def greedy_id(sim_matrix, id_prev, sig_iou = 0.4):
    """
    Compute the greedy algorithm to find the best id for each box.
    """
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

def display(boxes, labels, tacks, frame):
    """
    Display the boxes on the image.
    """
    image_name = "ADL-Rundle-6/img1/" + str(int(labels)).zfill(6) + ".jpg"
    img = cv2.imread(image_name)
    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][0]) + int(boxes[i][2]), int(boxes[i][1]) + int(boxes[i][3])), (0, 255, 0), 2)
        cv2.putText(img, str(tacks[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2, cv2.LINE_AA)
        cv2.putText(img, str(int(frame.iloc[i].iloc[6])), (int(boxes[i][0]) + int(boxes[i][2]) , int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,0), 2, cv2.LINE_AA)
    cv2.imshow('jsp',img)


sigma = 15 # threshold for the detection
det = pd.read_csv('ADL-Rundle-6/det/det.txt')
det_grouped_frame = det.groupby('frame')
nb_group = det_grouped_frame.ngroups    # number of frames
# init tracking
frame1 = det_grouped_frame.get_group(1)
boxes2 = frame_to_boxes(frame1,sigma)
tracks = update_tracks(0, [], len(frame1))
last_frame = frame1
for i in range(2,nb_group,1):
    ## display 
    display(boxes2, last_frame.iloc[0].iloc[0], tracks, last_frame)
    k = cv2.waitKey(0)
    # End if ESC pressed
    if k == 27:
        break
    act_frame = det_grouped_frame.get_group(i)
    # generate the boxes
    boxes1 = frame_to_boxes(last_frame,sigma)
    boxes2 = frame_to_boxes(act_frame,sigma)
    # compute the similarity matrix
    sim_matrix = similarity(boxes1, boxes2)
    # compute the greedy algorithm
    couples = greedy_id(sim_matrix, tracks)
    # update the tracks
    tracks = update_tracks(max(tracks), couples, len(act_frame))
    last_frame = act_frame