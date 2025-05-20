# System imports
import sys, os
import select
import ctypes, struct
import time
import tty
import termios
import argparse

# Package imports
import cv2

# Custom imports
from utils import *
from yolo_utils import *
from gps_utils import *
from MessageCenter import MessageCenter

# Given: face_roi is the cropped BGR image
def preprocess_face(face_roi):
    # Step 1: CLAHE for contrast
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    filtered = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    # Optional: add blur or sharpening here based on testing

    return filtered

def image_processing(message_center):
    frame = picam2.capture_array()
    if frame is None:
        print("[ERROR] Frame capture failed.")
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gender = face_processing(frame)
    message_center.add_face_detection(gender)

    # Object detection
    # objects: crosswalk, speedlimit, stop, trafficlight
    outputs = convert_to_blob(frame, network, 128, 128)
    bounding_boxes, class_objects, confidence_probs = object_detection(
        outputs, frame, 0.5
    )

    # sort the detected objets by confidence and only send the best 2 detections
    bounding_boxes, class_objects, confidence_probs = sort_by_confidence(
        2, confidence_probs, bounding_boxes, class_objects
    )

    if len(bounding_boxes) > 0:
        message_center.add_yolo_detection(
            class_objects[0], bounding_boxes[0], confidence_probs[0]
        )
        if class_objects[0] == 3:
            print(f"[INFO] Detected traffic light")
            
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            x, y, w, h = bounding_boxes[0]  # Assuming (x, y, w, h) format
            x, y, w, h = int(x), int(y), int(w), int(h)
            
                # Clip to image dimensions to avoid out-of-bounds errors
            height, width = hsv.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            # Sanity check: ensure region is non-empty
            if x2 > x1 and y2 > y1:
                hsv_roi = hsv[y1:y2, x1:x2]
            
            lower_red1 = np.array([170, 70, 50])
            upper_red1 = np.array([180, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            lower_green = np.array([70, 70, 50])
            upper_green = np.array([80, 255, 255])
            
            red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            red_mask = red_mask1 + red_mask2
            
            green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

            red_pixels = cv2.countNonZero(red_mask)
            green_pixels = cv2.countNonZero(green_mask)

            roi_area = hsv_roi.shape[0] * hsv_roi.shape[1]
            MIN_GREEN_PERCENT = 0.25
            min_green_pixels = roi_area * MIN_GREEN_PERCENT
            
            if green_pixels >= min_green_pixels and green_pixels > red_pixels:
                status = True
                print(f"[INFO] Green")

            else:
                
                status = False # red light =False, green light = True
                print(f"[INFO] Red")
            
            

            message_center.add_traffic_light(status)
        for box, cls, conf in zip(bounding_boxes, class_objects, confidence_probs):
            x, y, w, h = map(int, box)
            label = f"{cls}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put label above the box
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Optional: overlay traffic light status
        if class_objects and class_objects[0] == 3:  # traffic light class
            status_str = "Green" if status else "Red"
            cv2.putText(frame, f"Traffic Light: {status_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if not status else (0, 255, 0), 2)

        # Show the frame in a window
        cv2.imshow("Live View", frame)
        cv2.waitKey(1)  # Required for real-time display

    else:
        message_center.add_no_object_detected()

def face_processing(frame):
    
    #print(f"[DEBUG] frame.shape = {frame.shape}")
    if frame is None:
        print("[ERROR] Received empty frame in face_processing.")
        return 0

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print("[ERROR] Frame has invalid dimensions.")
        return 0
    h = frame.shape[0]
    w = frame.shape[1]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    face_detector = cv2.dnn.readNet('./cfg/opencv_face_detector_uint8.pb', './cfg/opencv_face_detector.pbtxt')
    gender_detector = cv2.dnn.readNet('./cfg/gender_net.caffemodel', './cfg/gender_deploy.prototxt')

    face_detector.setInput(blob)
    detections = face_detector.forward()

    # sort detections by confidence (from big to small)
    detections = sorted(detections[0, 0, :, :], key=lambda x: x[2], reverse=True)
    if len(detections) == 0:
        print("[WARN] No face detections found.")
        return 0
    detection = detections[0]




    # Check detection is well-formed
    if detection is None or len(detection) < 7:
        print("[WARN] Invalid detection format.")
        return 0

    confidence = detection[2]
    if confidence > 0.7:
        print("Face detected")
        faceBox = detection[3:7] * np.array([w, h, w, h])

        if None in faceBox or np.any(np.isnan(faceBox)):
            print("[WARN] Invalid face box values (None or NaN).")
            return 0

        faceBox = faceBox.astype("int")

        x1 = max(0, faceBox[0] - 15)
        y1 = max(0, faceBox[1] - 15)
        x2 = min(faceBox[2] + 15, frame.shape[1] - 1)
        y2 = min(faceBox[3] + 15, frame.shape[0] - 1)

        if x2 > x1 and y2 > y1:
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                prepfac = preprocess_face(face_roi)
                blob = cv2.dnn.blobFromImage(prepfac, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_detector.setInput(blob)
                genderPreds = gender_detector.forward()
                gender = genderPreds[0].argmax() + 1
                print(gender)
                return gender
            else:
                print("[WARN] Empty face ROI.")
        else:
            print("[WARN] Invalid face box dimensions.")
    #else:
        #print("[INFO] Detection below confidence threshold.")

    return 0

            
def gps_processing(message_center):
    data, addr = sock.recvfrom(1024)
    line = data.decode().strip()

    try:
        distances_m = list(map(float, line.split(",")))
    except ValueError:
        pass

    position, error = trilaterate_2D(distances_m)
    if position is not None:
        # print(f"[POS] x = {position[0]:.2f} ft, y = {position[1]:.2f} ft, RMSE = {error:.2f} ft")
        message_center.add_gps_position(position[0], position[1])

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="My program with options")
    parser.add_argument(
        "-gps",
        "--gps",
        action="store_true",
        default=False,
        help="Enable GPS in the program",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode, printing debug messages",
    )
    args = parser.parse_args()

    # initialize the camera and GPS
    camera_initialization()
    # initialize gps if enabled
    if args.gps:
        gps_initialization()

    # initialize the message center
    message_center = MessageCenter("/dev/ttyUSB0", 9600, args.debug)

    # Welcome message :)
    print_seal()

    # system start time
    program_start_time = time.time()
    time_stamp = program_start_time
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)  # or tty.setraw(fd)
        while True:
            image_processing(message_center)

            if args.gps:
                gps_processing(message_center)

            # process messages
            message_center.processing_tick()

            # handle user input
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                # print(f"You typed: {ch}")
                if ch == "q":
                    print("\nUser quit command detected. Exiting program...")
                    break
                else:
                    print(f"\nUnrecognized command: {ch}")

            # delay for a short period to avoid busy waiting
            time.sleep(0.1)

            # print time elapsed since start and time interval without newline
            interval = (time.time() - time_stamp) * 1000
            time_stamp = time.time()
            print(
                f"\rTime elapsed: {get_time_millis(program_start_time):.2f} ms; loop interval: {interval:.02f} ms",
                end="",
            )
            print("\b" * 40, end="")  # Clear the line

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        # do some cleanup if necessary

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return


if __name__ == "__main__":
    main()
