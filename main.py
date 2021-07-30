# Import the necessary packages
from imutils import face_utils
import dlib
import cv2
import glob
import os
import time
import numpy as np
from subprocess import call
from colored import fg, bg, attr

import settings

# Initialize dlib's face detector (HOG-based) and then create
# The facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


# Functions start here

def clear():
    # check and make call for specific operating system
    _ = call('clear' if os.name =='posix' else 'cls')

def detect_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image

        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        collection =  [         
          {"tag": "Jaw",
          "shape": shape[0:17],
          "color": (255, 0, 0),
          "size": 1,
          "closed":False},

          {"tag": "Eyebrow left",
          "shape": shape[17:22],
          "color": (255, 0, 0),
          "size": 1,
          "closed":False},

          {"tag": "Eyebrow right",
          "shape": shape[22:27],
          "color": (255, 0, 0),
          "size": 1,
          "closed":False},

          {"tag": "Nose Line",
          "shape": shape[27:31],
          "color": (255, 0, 0),
          "size": 1,
          "closed":False},

          {"tag": "Nose curve",
          "shape": shape[31:36],
          "color": (255, 0, 0),
          "size": 1,
          "closed":False},
          
          {"tag": "Left Eye",
          "shape": shape[36:42],
          "color": (255, 0, 0),
          "size": 1,
          "closed":True},
          
          {"tag": "Right Eye",
          "shape": shape[42:48],
          "color": (255, 0, 0),
          "size": 1,
          "closed":True},

          {"tag": "Mouth outer",
          "shape": shape[48:60],
          "color": (255, 0, 0),
          "size": 1,
          "closed":True},

          {"tag": "Mouth inner",
          "shape": shape[60:68],
          "color": (255, 0, 0),
          "size": 1,
          "closed":True},
        ]
        if settings.Join_the_lines == True:
            # Drawing polylines in image
            # i.e joining the facial features points to form lines
            for i in collection :
                pts = np.array(i['shape'],np.int32) 
                pts = pts.reshape((-1, 1, 2))
                isClosed = i['closed']
                color = i['color']
                thickness = i['size']      
                image = cv2.polylines(image, [pts], isClosed, color, thickness)
    return image

def detect_edges(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      
    # define range of red color in HSV
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
      
    # create a red HSV colour boundary and 
    # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)
  
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    edges = cv2.Canny(image,100,200)

    return edges

def choice_1():
    folder = input("Enter name of folder in which image are present or press enter to use default\nBy default it will take 'face_images' folder: ")
    
    print('%s Note: You need to press esc button to escape from escape from video stream %s' % (fg(11), attr(0)))
    input("Press any button to continue")

    if len(folder) == 0:
        folder = "face_images"

    # Batch proccessing images
    # fn = file name

    for fn in glob.glob(folder + '/*.' + settings.Image_extension):
        original = cv2.imread(fn, cv2.IMREAD_COLOR)
        new = detect_face_landmarks(original)
        cv2.imshow("Image name : " + fn,new)
        while True:
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()        

def choice_2():
    cap = cv2.VideoCapture(0)
    print('%s Note: You need to press esc button to escape from escape from video stream %s' % (fg(11), attr(0)))
    input("Press any button to continue")
    while True:
        _, original = cap.read()
        new = detect_face_landmarks(original)
        cv2.imshow("Webcam stream",new)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()        

def choice_3():
    folder = input("Enter name of folder in which image are present or press enter to use default\nBy default it will take 'monuments_images' folder: ")
    print('%s Note: You need to press esc button to escape from escape from image %s' % (fg(11), attr(0)))
    input("Press any button to continue")
    if len(folder) == 0:
        folder = "monuments_images"

    # Batch proccessing images
    # fn means file name

    for fn in glob.glob(folder + '/*.' + settings.Image_extension):
        original = cv2.imread(fn, cv2.IMREAD_COLOR)
        new = detect_edges(original)
        cv2.imshow("Image name : " + fn,new)
        while True:
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows() 

def choice_4():
    cap = cv2.VideoCapture(0)
    print('%s Note: You need to press esc button to escape from video stream %s' % (fg(11), attr(0)))
    input("Press any button to continue")
    while True:
        _, original = cap.read()
        new = detect_edges(original)
        cv2.imshow("Webcam stream",new)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()   

def main():
    string = ".----------------.  .----------------.  .----------------.  .----------------. \n| .--------------. || .--------------. || .--------------. || .--------------. |\n| |     ______   | || |     ____     | || |  _________   | || | ____    ____ | |\n| |   .' ___  |  | || |   .'    `.   | || | |  _   _  |  | || ||_   \\  /   _|| |\n| |  / .'   \\_|  | || |  /  .--.  \\  | || | |_/ | | \\_|  | || |  |   \\/   |  | |\n| |  | |         | || |  | |    | |  | || |     | |      | || |  | |\\  /| |  | |\n| |  \\ `.___.'\\  | || |  \\  `--'  /  | || |    _| |_     | || | _| |_\\/_| |_ | |\n| |   `._____.'  | || |   `.____.'   | || |   |_____|    | || ||_____||_____|| |\n| |              | || |              | || |              | || |              | |\n| '--------------' || '--------------' || '--------------' || '--------------' |\n '----------------'  '----------------'  '----------------'  '----------------' \n"
    while True:
        if settings.Clear_terminal == True:
            clear()
        print(string)
        print("Welcome to submission for COTM")
        print("What would you like to do?")

        print("(1) Detect face in images")
        print("(2) Detect face in webcam")
        print("(3) Detect edges in images")
        print("(4) Detect edges in webcam")
        print("(5) Quit")
        print("(6) Settings")
        print("\n")

        choice = input("Enter the number : ")

        if choice == '1':
            choice_1()
        elif choice == '2':
            choice_2()
        elif choice == '3':
            choice_3()
        elif choice == '4':
            choice_4()
        elif choice == '5':
            break
        elif choice == '6':
            print("Trying to open settings in VS Code")
            os.system('code settings.py')
            break
        else:
            print('%s Please enter a valid option %s' % (fg(11), attr(0)))
            time.sleep(1)

# main()
try:
    main()
except:
    print('%s%s Something went wrong %s' % (fg(1), bg(15), attr(0)))
print("Bye-Bye")
print("Have a nice day ahead")