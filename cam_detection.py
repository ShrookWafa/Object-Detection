import numpy as np
import cv2
from imageai.Detection import ObjectDetection
import os
import colorsys
import random

# Generate random n colors
def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed='fastest')

cap = cv2.VideoCapture(0)
save_count = 0

# Define a color for every class
color_count = 0
color_map = dict()

# Generate 80 colors for bounding boxes and shuffle them
colors = get_colors(80)
for i in range(80):
    colors[i] = tuple(255*x for x in colors[i])
random.shuffle(colors)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('temp.jpg', frame)
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "temp.jpg"))

    # Draw bounding boxes around all detections
    for i in range(len(detections)):
        n = detections[i]['name']
        if n not in color_map:
            color_map[n] = color_count
            color_count += 1
        
        box = detections[i]['box_points']
        c = colors[color_map[n]]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), c, 2)
        cv2.putText(frame, n,(box[0]-3, box[1]-3), 2, 1.5, c)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    # to quit camera press q
    if key & 0xFF == ord('q'):
        break
    # to save current frame press s
    elif key & 0xFF == ord('s'):
        s = 'image' + str(save_count) + '.jpg'
        save_count += 1
        cv2.imwrite(s, frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
os.remove('temp.jpg')
