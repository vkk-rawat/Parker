import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
import time
from datetime import datetime
import json
import threading
from queue import Queue


class ParkingSpaceMonitor:
    def __init__(self, api_key, stream_url=0, confidence=0.5):
        """
        Initialize the parking space monitor

        Args:
            api_key (str): Roboflow API key
            stream_url: Camera index or RTSP/HTTP stream URL
            confidence (float): Detection confidence threshold
        """
        self.stream_url = stream_url
        self.confidence = confidence
        self.running = False
        self.frame_queue = Queue(maxsize=30)

        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        self.model = rf.workspace().project("vehicle-detection").version(1).model

        # Initialize annotator
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        # Parking spaces format: {id: {"coords": [x1,y1,x2,y2], "status": "free/occupied"}}
        self.parking_spaces = {}

        # Status change callback
        self.status_callback = None

    def add_parking_space(self, space_id, coordinates):
        """
        Add a parking space to monitor

        Args:
            space_id (str): Unique identifier for the parking space
            coordinates (list): [x1,y1,x2,y2] coordinates of parking space
        """
        self.parking_spaces[space_id] = {
            "coords": coordinates,
            "status": "free"
        }

    def set_status_callback(self, callback):
        """Set callback function for status changes"""
        self.status_callback = callback

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / (area1 + area2 - intersection)

    def check_parking_spaces(self, detections):
        """Check if parking spaces are occupied based on detections"""
        status_changed = False

        for space_id, space in self.parking_spaces.items():
            space_box = space["coords"]
            is_occupied = False

            for detection in detections:
                # 30% overlap threshold
                if self.calculate_iou(space_box, detection[:4]) > 0.3:
                    is_occupied = True
                    break

            new_status = "occupied" if is_occupied else "free"
            if new_status != space["status"]:
                space["status"] = new_status
                status_changed = True

        if status_changed and self.status_callback:
            self.status_callback(self.get_parking_status())

    def get_parking_status(self):
        """Get current status of all parking spaces"""
        return {
            space_id: space["status"]
            for space_id, space in self.parking_spaces.items()
        }

    def start_monitoring(self):
        """Start the video stream and monitoring"""
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_reader)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def _stream_reader(self):
        """Read frames from the video stream"""
        cap = cv2.VideoCapture(self.stream_url)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame, attempting to reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.stream_url)
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        cap.release()

    def process_frames(self):
        """Process frames and monitor parking spaces"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # Make prediction
                results = self.model.predict(
                    frame, confidence=self.confidence).json()

                # Extract detections
                detections = []
                for prediction in results['predictions']:
                    x1 = prediction['x'] - prediction['width'] / 2
                    y1 = prediction['y'] - prediction['height'] / 2
                    x2 = prediction['x'] + prediction['width'] / 2
                    y2 = prediction['y'] + prediction['height'] / 2
                    confidence = prediction['confidence']
                    class_name = prediction['class']

                    detections.append([x1, y1, x2, y2, confidence, class_name])

                # Check parking space status
                self.check_parking_spaces(detections)

                # Draw parking spaces and detections
                frame_annotated = frame.copy()

                # Draw parking spaces
                for space_id, space in self.parking_spaces.items():
                    coords = space["coords"]
                    color = (0, 255, 0) if space["status"] == "free" else (
                        0, 0, 255)
                    cv2.rectangle(frame_annotated,
                                  (int(coords[0]), int(coords[1])),
                                  (int(coords[2]), int(coords[3])),
                                  color, 2)
                    cv2.putText(frame_annotated,
                                f"Space {space_id}: {space['status']}",
                                (int(coords[0]), int(coords[1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw vehicle detections
                frame_annotated = self.box_annotator.annotate(
                    scene=frame_annotated,
                    detections=detections,
                    labels=[f"{d[5]}: {d[4]:.2f}" for d in detections]
                )

                # Display frame
                cv2.imshow('Parking Space Monitor', frame_annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

    def stop(self):
        """Stop monitoring and release resources"""
        self.running = False
        if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
            self.stream_thread.join()
        cv2.destroyAllWindows()

# Example status change callback


def status_changed(status):
    print(f"Parking status updated: {json.dumps(status, indent=2)}")


def main():
    # Replace with your Roboflow API key
    API_KEY = "vMt0JUmSrXKPCmTa9jOO"

    # Create monitor
    monitor = ParkingSpaceMonitor(API_KEY)

    # Add parking spaces to monitor
    # Replace with actual coordinates from your camera view
    monitor.add_parking_space("A1", [100, 100, 300, 300])
    monitor.add_parking_space("A2", [350, 100, 550, 300])
    monitor.add_parking_space("B1", [100, 350, 300, 550])

    # Set status change callback
    monitor.set_status_callback(status_changed)

    try:
        monitor.start_monitoring()
        monitor.process_frames()
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
