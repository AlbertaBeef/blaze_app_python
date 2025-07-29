
import numpy as np
import cv2
#import torch

def draw_detections(img, detections, with_keypoints=True):
    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]
        
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1) 

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2    ])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)


def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)



def draw_landmarks(img, points, connections=[], color=(0, 255, 0), size=2):
    points = points[:,:2]
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=size)
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(img, (x0, y0), (x1, y1), (0,0,0), size)



# https://github.com/metalwhale/hand_tracking/blob/b2a650d61b4ab917a2367a05b85765b81c0564f2/run.py
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

POSE_UPPER_BODY_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,13), (13,15), (15,17), (17,19), (19,15), (15,21),
    (12,14), (14,16), (16,18), (18,20), (20,16), (16,22),
    (11,12), (12,24), (24,23), (23,11)
]
POSE_FULL_BODY_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,13), (13,15), (15,17), (17,19), (19,15), (15,21),
    (12,14), (14,16), (16,18), (18,20), (20,16), (16,22),
    (11,12), (12,24), (24,23), (23,11),
    (24,26), (26,28), (28,30), (30,32), (32,28),
    (23,25), (25,27), (27,29), (29,31), (31,27)
]

# Vertex indices can be found in
# github.com/google/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualisation.png
# Found in github.com/google/mediapipe/python/solutions/face_mesh.py
FACE_CONNECTIONS = [
    # Lips.
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye.
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
    (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293),
    (293, 334), (334, 296), (296, 336),
    # Right eye.
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
    (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105),
    (105, 66), (66, 107),
    # Face oval.
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
    (109, 10)
]



def draw_detection_scores(detection_scores, min_score_thresh):
    num_anchors = detection_scores.shape[1]
    
    x = range(num_anchors)
    y = detection_scores[0,:]

    plot = np.zeros((500,500))
    xdiv = int((num_anchors / 500)+1)
    for i in range(1,num_anchors):
        x1 = int((i-1)/xdiv);
        y1 = int(500 - y[i-1]*500);
        x2 = int((i)/xdiv);
        y2 = int(500 - y[i]*500);
        cv2.line(plot, (x1,y1), (x2,y2), 255, 1);

    # draw threshold level
    x1=0
    x2=499
    y1=int(500-min_score_thresh*500)
    y2=y1
    cv2.line(plot, (x1,y1), (x2,y2), 255, 1);
        
    #cv2.imshow("Detection Scores (sigmoid)",plot)
    return plot


# Colors for each bar (BGR)
stacked_bar_generic_colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 128, 128), # Gray
    (0, 128, 255),   # Orange
    (0, 0, 0),       # Black
    (255, 255, 255)  # White
]

stacked_bar_latency_colors = [
    (255,   0,   0), # resize         : blue 
    (0,   255,   0), # detector_pre   : green
    (255,   0, 255), # detector_model : magenta
    (255, 255,   0), # detector_post  : cyan
    (255,   0,   0), # extract_roi    : blue
    (0,   255,   0), # landmark_pre   : green
    (255,   0, 255), # landmark_model : magenta
    (255, 255,   0), # landmark_post  : cyan
    (0,     0,   0), # annotate       : black
]

stacked_bar_performance_colors = [
    (255, 0, 255),  # pipeline_fps : magenta
]

# Example usage for labels
#component_labels = [
#    "resize",
#    "detector[pre]",
#    "detector[model]",
#    "detector[post]",
#    "extract_roi",
#    "landmark[pre]",
#    "landmark[model]",
#    "landmark[post]",
#    "annotate"
#]

def draw_stacked_bar_chart(
    pipeline_titles,
    component_labels,
    component_values,  # [component][pipeline]
    component_colors,
    chart_name
):
    pipelines = len(pipeline_titles)
    components = len(component_labels)

    # Find max stacked bar value (sum of components for each pipeline)
    max_stacked = 0.0
    for i in range(pipelines):
        sum_val = sum(component_values[j][i] for j in range(components))
        if sum_val > max_stacked:
            max_stacked = sum_val

    # Chart size
    chart_width = 800
    legend_spacing = 10
    max_legend_per_line = 4
    legend_lines = (components + max_legend_per_line - 1) // max_legend_per_line
    legend_line_height = 28
    chart_height = 40 * pipelines + 80 + legend_spacing + legend_lines * legend_line_height
    left_margin = 160
    bar_height = 28
    spacing = 12

    chart = np.full((chart_height, chart_width, 3), 255, dtype=np.uint8)
    cv2.putText(chart, chart_name, (left_margin, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,40,40), 2, cv2.LINE_AA)

    # Draw y labels (pipeline names)
    for i in range(pipelines):
        y = 60 + i * (bar_height + spacing) + bar_height//2 + 5
        cv2.putText(chart, pipeline_titles[i], (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,40,40), 1, cv2.LINE_AA)

    # Draw bars (stacked, normalized so max stacked bar fits in chart)
    for i in range(pipelines):
        y = 60 + i * (bar_height + spacing)
        x = left_margin
        sum_val = sum(component_values[j][i] for j in range(components))
        norm_factor = ((chart_width - left_margin - 100) / max_stacked) if max_stacked > 0.0 else 0.0
        x_local = x
        for j in range(components):
            val = component_values[j][i]
            bar_w = int(val * norm_factor) if norm_factor > 0.0 else 0
            if bar_w > 0:
                start_point = (x_local, y)
                end_point = (x_local+bar_w, y+bar_height)
                cv2.rectangle(chart, start_point, end_point, component_colors[j], -1)
                # Optionally draw value
                if val >= 0.001:
                    cv2.putText(chart, f"{val:.3f}", (x_local+4, y+bar_height-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            x_local += bar_w
        # draw total
        cv2.putText(chart, f"{sum_val:.3f}", (x_local+4, y+bar_height-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,40,40), 1, cv2.LINE_AA)

    # Draw legend on multiple lines, max 4 per line
    legend_start_x = left_margin
    legend_start_y = chart_height - legend_lines * legend_line_height + 6
    legend_item_width = (chart_width - left_margin - 30) // max_legend_per_line
    for line in range(legend_lines):
        leg_y = legend_start_y + line * legend_line_height
        for j in range(max_legend_per_line):
            idx = line * max_legend_per_line + j
            if idx >= components:
                break
            leg_x = legend_start_x + j * legend_item_width
            start_point = (leg_x, leg_y)
            end_point = (leg_x+20, leg_y+18)            
            cv2.rectangle(chart, start_point, end_point, component_colors[idx], -1)
            cv2.putText(chart, component_labels[idx], (leg_x + 28, leg_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,40,40), 1, cv2.LINE_AA)

    return chart
