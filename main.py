import os.path
import cv2
import mediapipe as mp
import argparse
from image_processing import process_img


args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default=None)

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Detect faces
mp_face_det = mp.solutions.face_detection

with mp_face_det.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode in ['image']:
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)

        # Save image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        while ret:

            frame = process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            if cv2.waitKey(100) == ord('q'):
                break
            ret, frame = cap.read()

        cap.release()