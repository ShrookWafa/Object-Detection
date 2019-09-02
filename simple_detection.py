from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed='fast')
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image2.jpeg"), 
                                             output_image_path=os.path.join(execution_path , "imagenew2.jpg"))

for i in range(len(detections)):
    print(detections[i])
