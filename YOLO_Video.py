from ultralytics import YOLO
import cv2
import math
import numpy as np
import json
import base64
import time
FOCAL_LENGTH = 800  # example value, this needs to be calibrated for your camera
CAR_WIDTH = 1.8  # average width of a car
TRUCK_WIDTH = 2.5  # average width of a truck

def calculate_distance(focal_length, real_width, pixel_width):
    return (real_width * focal_length) / pixel_width

def warn_if_close(distance, threshold=1.3):  # threshold distance in meters
    return distance < threshold
def standardize_input(image):
    image_crop = np.copy(image)
    row_crop = 7
    col_crop = 8
    if image_crop.shape[0] > 2 * row_crop and image_crop.shape[1] > 2 * col_crop:
        image_crop = image[row_crop:-row_crop, col_crop:-col_crop, :]
        standard_im = cv2.resize(image_crop, (32, 32))
        return standard_im
    return None


def create_feature_crop_red(rgb_image):
    image_crop_red = rgb_image[:13, 10:-10, :]
    if image_crop_red.shape[0] > 0 and image_crop_red.shape[1] > 0:
        image_crop_red_10x10 = cv2.resize(image_crop_red, (10, 10))
        hsv_crop_red = cv2.cvtColor(image_crop_red_10x10, cv2.COLOR_RGB2HSV)
        feature = np.sum(hsv_crop_red[:, :, 2])
        return feature
    return 0


def create_feature_crop_yellow(rgb_image):
    image_crop_yellow = rgb_image[12:21, 10:-10, :]
    if image_crop_yellow.shape[0] > 0 and image_crop_yellow.shape[1] > 0:
        image_crop_yellow_10x10 = cv2.resize(image_crop_yellow, (10, 10))
        hsv_crop_yellow = cv2.cvtColor(image_crop_yellow_10x10, cv2.COLOR_RGB2HSV)
        feature = np.sum(hsv_crop_yellow[:, :, 2])
        return feature
    return 0


def create_feature_crop_green(rgb_image):
    image_crop_green = rgb_image[21:, 10:-10, :]
    if image_crop_green.shape[0] > 0 and image_crop_green.shape[1] > 0:
        image_crop_green_10x10 = cv2.resize(image_crop_green, (10, 10))
        hsv_crop_green = cv2.cvtColor(image_crop_green_10x10, cv2.COLOR_RGB2HSV)
        feature = np.sum(hsv_crop_green[:, :, 2])
        return feature
    return 0

def is_brightness_sufficient(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = np.mean(hsv_image[:, :, 2])
    return brightness > 50  # Threshold value to be adjusted based on experimentation
def classify_traffic_light_color(traffic_light):
    standardized_light = standardize_input(traffic_light)
    if standardized_light is None:
        return "unknown"
    if not is_brightness_sufficient(standardized_light):
        return "unknown"
    hsv = cv2.cvtColor(standardized_light, cv2.COLOR_RGB2HSV)
    red_feature = create_feature_crop_red(hsv)
    yellow_feature = create_feature_crop_yellow(hsv)
    green_feature = create_feature_crop_green(hsv)

    if red_feature > yellow_feature and red_feature > green_feature:
        return "red" if red_feature > 0.8 * (yellow_feature + green_feature) else "unknown"
    elif yellow_feature > red_feature and yellow_feature > green_feature:
        return "yellow" if yellow_feature > 0.7 * (red_feature + green_feature) else "unknown"
    elif green_feature > red_feature and green_feature > yellow_feature:
        return "green" if green_feature > 0.8 * (red_feature + yellow_feature) else "unknown"
    else:
        return "unknown"
    # if red_feature > yellow_feature and red_feature > green_feature:
    #     return "red"
    # elif yellow_feature > red_feature and yellow_feature > green_feature:
    #     return "yellow"
    # elif green_feature > red_feature and green_feature > yellow_feature:
    #     return "green"
    # else:
    #     return "unknown"

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    if not cap.isOpened():
        raise ValueError(f"Video file {path_x} cannot be opened")
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    model=YOLO("yolov8n.pt")
    # model=YOLO("E:/ĐATN/Livestream/YOLOv8-CrashCourse/Weights/best.pt")
    # classNames = [
    #               "traffic light",
    #               "traffic sign",
    #               "car",
    #               "person",
    #               "truck",
    #             ]
    classNames = [
                  "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                ]
    start_time = time.time()
    frame_count = 0
    while True:
        success, img = cap.read()
        # if not success:
        #     break
        # frame_count += 1
        # if frame_count % frame_skip != 0:
        #     continue
        # img = resize_frame(img, width=int(frame_width * 0.5))
        # img = resize_frame(img, width=int(frame_width * 0.3), height=int(frame_height * 0.3))
        traffic_light_color = "none"
        results=model(img,stream=True)
        detection_data = []
        start_time = time.time()  # Lấy thời gian bắt đầu nhận diện
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                # print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                width_in_pixels = x2 - x1
                if class_name == "car" or class_name == "truck":
                    real_width = CAR_WIDTH if class_name == "car" else TRUCK_WIDTH
                    distance = calculate_distance(FOCAL_LENGTH, real_width, width_in_pixels)
                    is_close = warn_if_close(distance)
                    if is_close:
                        detection_data.append({
                            "object": class_name,
                            "distance": distance,
                            "warning": f"Warning: {class_name} is too close!"
                        })
                    else:
                        detection_data.append({
                            "object": class_name,
                            "distance": distance
                        })
                if class_name == "traffic light":
                    # Cắt hình ảnh đèn giao thông trước khi vẽ bounding box
                    traffic_light = img[y1:y2, x1:x2]
                    color = classify_traffic_light_color(traffic_light)
                    traffic_light_color=color
                    label = f'{color} {conf}'
                    detection_data.append({"object": "traffic light", "color": color})
                else:
                    label = f'{class_name} {conf}'
                    detection_data.append({"object": class_name, "color": None})
                # label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                # print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        # yield img,traffic_light_color
        detection_data.append(({"time" : int((time.time() - start_time) * 1000)}))
        ret, buffer = cv2.imencode('.jpg', img)
        frame = base64.b64encode(buffer).decode('utf-8')
        event_data = f"data: {frame}\nevent: image\n\n"
        for detection in detection_data:
            detection_json = json.dumps(detection)
            event_data += f"data: {detection_json}\nevent: detection\n\n"
        yield event_data
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()