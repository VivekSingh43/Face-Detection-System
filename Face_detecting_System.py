import cv2
import numpy as np
import os
from datetime import datetime
import argparse

class FaceDetectionSystem:
    def __init__(self):
        # Initialize Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize DNN face detector (more accurate)
        self.load_dnn_model()
        
        # Statistics
        self.total_faces_detected = 0
        self.detection_method = "haar"  # "haar" or "dnn"
        
        # Create output directory for screenshots
        self.output_dir = "detected_faces"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_dnn_model(self):
        """Load pre-trained DNN model for face detection"""
        try:
            # You would need to download these files for DNN detection
            # For now, we'll use Haar cascades as the primary method
            self.net = None
            print("DNN model not loaded. Using Haar Cascade method.")
        except:
            self.net = None
            print("DNN model not available. Using Haar Cascade method.")
    
    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces, gray
    
    def detect_faces_dnn(self, frame):
        """Detect faces using DNN method (placeholder - requires model files)"""
        # This would be implemented with actual DNN model files
        # For demo purposes, falling back to Haar cascade
        return self.detect_faces_haar(frame)
    
    def draw_detections(self, frame, faces, gray=None):
        """Draw bounding boxes around detected faces"""
        face_count = 0
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add face number label
            face_count += 1
            cv2.putText(frame, f'Face {face_count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Optional: Detect eyes within face region
            if gray is not None:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)
        
        # Display statistics
        cv2.putText(frame, f'Faces Detected: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Detected: {self.total_faces_detected}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def save_detected_faces(self, frame, faces):
        """Save individual detected faces as separate images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Save face image
            filename = f"{self.output_dir}/face_{timestamp}_{i+1}.jpg"
            cv2.imwrite(filename, face_img)
            print(f"Saved face: {filename}")
    
    def detect_from_webcam(self):
        """Real-time face detection from webcam"""
        print("Starting webcam face detection...")
        print("Press 's' to save detected faces, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            if self.detection_method == "haar":
                faces, gray = self.detect_faces_haar(frame)
            else:
                faces, gray = self.detect_faces_dnn(frame)
            
            # Update statistics
            if len(faces) > 0:
                self.total_faces_detected += len(faces)
            
            # Draw detections
            frame = self.draw_detections(frame, faces, gray)
            
            # Display frame
            cv2.imshow('Face Detection System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) > 0:
                self.save_detected_faces(frame, faces)
                print("Faces saved!")
            elif key == ord('h'):
                self.detection_method = "haar"
                print("Switched to Haar Cascade detection")
            elif key == ord('d'):
                self.detection_method = "dnn"
                print("Switched to DNN detection")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Session ended. Total faces detected: {self.total_faces_detected}")
    
    def detect_from_image(self, image_path):
        """Detect faces from a single image"""
        print(f"Processing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not load image")
            return
        
        # Detect faces
        if self.detection_method == "haar":
            faces, gray = self.detect_faces_haar(frame)
        else:
            faces, gray = self.detect_faces_dnn(frame)
        
        # Draw detections
        result_frame = self.draw_detections(frame, faces, gray)
        
        # Save result
        output_path = f"{self.output_dir}/result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_frame)
        print(f"Result saved: {output_path}")
        
        # Display result
        cv2.imshow('Face Detection Result', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"Detected {len(faces)} faces in the image")
        return len(faces)
    
    def detect_from_video(self, video_path):
        """Detect faces from a video file"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"{self.output_dir}/result_{os.path.basename(video_path)}"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_faces = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            if self.detection_method == "haar":
                faces, gray = self.detect_faces_haar(frame)
            else:
                faces, gray = self.detect_faces_dnn(frame)
            
            total_faces += len(faces)
            
            # Draw detections
            result_frame = self.draw_detections(frame, faces, gray)
            
            # Write frame to output video
            out.write(result_frame)
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, detected {total_faces} faces so far")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total faces detected: {total_faces}")
        print(f"Output saved: {output_path}")
    
    def get_statistics(self):
        """Return detection statistics"""
        return {
            "total_faces_detected": self.total_faces_detected,
            "detection_method": self.detection_method,
            "output_directory": self.output_dir
        }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Face Detection System using OpenCV')
    parser.add_argument('--mode', choices=['webcam', 'image', 'video'], default='webcam', help='Detection mode')
    parser.add_argument('--input', help='Input file path for image/video mode')
    parser.add_argument('--method', choices=['haar', 'dnn'], default='haar', help='Detection method')
    
    args = parser.parse_args()
    
    # Initialize face detection system
    detector = FaceDetectionSystem()
    detector.detection_method = args.method
    
    print("=" * 50)
    print("FACE DETECTION SYSTEM")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Method: {args.method}")
    print("=" * 50)
    
    try:
        if args.mode == 'webcam':
            detector.detect_from_webcam()
        elif args.mode == 'image':
            if not args.input:
                print("Error: Please provide input image path with --input")
                return
            detector.detect_from_image(args.input)
        elif args.mode == 'video':
            if not args.input:
                print("Error: Please provide input video path with --input")
                return
            detector.detect_from_video(args.input)
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    
    # Display final statistics
    stats = detector.get_statistics()
    print("\n" + "=" * 50)
    print("SESSION STATISTICS")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()