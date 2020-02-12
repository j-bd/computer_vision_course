#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:18:31 2020

@author: j-bd
"""

import argparse

import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def arguments_parser():
    '''Retrieve  user data command'''
    parser = argparse.ArgumentParser(
        prog="Smile Detection",
        usage='''%(prog)s [Detection procedure]''',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch execution:
        -------------------------------------
        python3 detect_smile.py
        --cascade "path/to/cascade/directory" --model "path/to/model"
        --video "path/to/video"

        The two first arguments are mandatory.
        '''
    )
    parser.add_argument(
        "-c", "--cascade", required=True,
        help="path to where the face cascade resides"
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="path to pre-trained smile detector CNN"
    )
    parser.add_argument(
        "-v", "--video", help="path to the (optional) video file"
    )
    args = vars(parser.parse_args())
    return args

def initialisation(args):
    '''Select detector, model and medium'''
    detector = cv2.CascadeClassifier(args["cascade"])
    model = load_model(args["model"])
    # If a video path was not supplied, grab the reference to the webcam
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    # Otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])
    return detector, model, camera

def detection(args, detector, model, camera):
    '''Medium analysis'''
    while True:
        # Grab the current frame
        (grabbed, frame) = camera.read()
        # If we are viewing a video and we did not grab a frame, then we have
        # reached the end of the video
        if args.get("video") and not grabbed:
            break
        # Resize the frame, convert it to grayscale, and then clone the original
        # frame so we can draw on it later in the program
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_clone = frame.copy()

        # Detect faces in the input frame, then clone the frame so that we can
        # draw on it
        face_bound_box = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Loop over the face bounding boxes
        for (x_val, y_val, w_val, h_val) in face_bound_box:
            # Extract the ROI of the face from the grayscale image, resize it to
            # a fixed 28x28 pixels, and then prepare the ROI for classification
            # via the CNN
            roi = gray[y_val:y_val + h_val, x_val:x_val + w_val]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Determine the probabilities of both "smiling" and "not smiling",
            # then set the label accordingly
            (notSmiling, smiling) = model.predict(roi)[0]
            label = "Smiling" if smiling > notSmiling else "Not Smiling"

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(
                frame_clone, label, (x_val, y_val + h_val + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
            )
            cv2.rectangle(
                frame_clone, (x_val, y_val), (x_val + w_val, y_val + h_val),
                (0, 0, 255), 2
            )

        # Show our detected faces along with smiling/not smiling labels
        cv2.imshow("Face", frame_clone)
        # If the ’q’ key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def main():
    '''Launch main steps'''
    args = arguments_parser()

    detector, model, camera = initialisation(args)

    detection(args, detector, model, camera)

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
