import cv2
import os
import pickle
import numpy as np
import pickle
import dlib as dlib
import math

from PIL import Image
import matplotlib.pyplot as plt
import keras.utils as image
from keras_vggface import utils
# from tensorflow.keras.utils import load_img, img_to_array

from tensorflow.keras.models import load_model

def shape_to_np(shape, dtype = "int"):
	coords = np.zeros((68, 2), dtype = dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords









def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):

    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))	
    #  print("")
    # print("")
    # print("pointd")
    # print("")
    # print(point_2d)
    return point_2d


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=100, rear_depth=0, front_size=500, front_depth=100,
                        color=(255, 255, 0), line_width=2):
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    return (x, y)


def get_angles_eular(rvec, tvec):
    rmat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rmat, tvec)) # projection matrix [R | t]
    degrees = -cv2.decomposeProjectionMatrix(P)[6]
    rx, ry, rz = degrees[:, 0]
    return [rx, ry, rz]

def get_angles_gerneal(image_points, rvec, tvec, CAMERA_MATRIX, img):
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rvec, tvec, CAMERA_MATRIX, dist_coeffs, cv2.SOLVEPNP_AP3P)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))

    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(img, p1, p2, (0, 255, 0), 2)

    m = (p2[1] - p1[1])/(p2[0] - p1[0])
    vertical_angle = int(math.degrees(math.atan(m)))

    x1, x2 = head_pose_points(img, rvec, tvec, CAMERA_MATRIX)

    cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

    if (x2[0] - x1[0]) != 0:
        m = (x2[1] - x1[1])/(x2[0] - x1[0])
        horizon_angle = int(math.degrees(math.atan(-1/m)))
    else: 
        horizon_angle = 0

    return (horizon_angle, vertical_angle)










FACE_DETECTOR = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# FACE_DETECTOR = dlib.get_frontal_face_detector()

FACE_PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

FACE_3D_MODEL_POINT = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ], dtype='double')


# dimension of images
image_width = 224
image_height = 224

def rect_to_bb(rect):
    x = rect.rect.left()
    y = rect.rect.top()
    w = rect.rect.right() - x
    h = rect.rect.bottom() - y
    return (x, y, w, h)

# def rect_to_bb(rect):
#     x = rect.left()
#     y = rect.top()
#     w = rect.right() - x
#     h = rect.bottom() - y
#     return (x, y, w, h)


# for detecting faces






for i in range(1,4): 
    test_image_filename = f'./facetest/face{i}.jpg'

    imgtest = cv2.imread(test_image_filename, cv2.IMREAD_COLOR)
    image_array = np.array(imgtest, "uint8")

    faces = FACE_DETECTOR(imgtest, 0)

    if len(faces) != 1: 
        print(f'---We need exactly 1 face;photo skipped---')
        print()
        continue

    for face in faces:

        (x_, y_, w, h) = rect_to_bb(face)
        # draw the face detected
        face_detect = cv2.rectangle(
            imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
        plt.imshow(face_detect)
        plt.show()
        
        
        marks = FACE_PREDICTOR(imgtest, face.rect)    # Use dlib

        marks = shape_to_np(marks)
            
        landmark = [marks[30], marks[8], marks[36], marks[45], marks[48], marks[54]]

        image_points = np.array([
                landmark[0],     # nose point
                landmark[1],     # chin point
                landmark[2],     # left corner of left eye
                landmark[3],     # right corner of right eye
                landmark[4],     # left corner of mouth
                landmark[5]      # right corner of mouth
        ], 
        dtype='double')
        
        
        
        
        
        size = imgtest.shape
        focal_length = size[1]
        print(focal_length)
        center = (size[1]/2, size[0]/2)
        CAMERA_MATRIX = np.array(
                            [
                                [focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]
                            ], dtype='double'
                        )
        
        print(CAMERA_MATRIX)
        
        dist_coeffs = np.zeros((4,1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(FACE_3D_MODEL_POINT, image_points, CAMERA_MATRIX, dist_coeffs)
        draw_annotation_box(imgtest, rotation_vector, translation_vector, CAMERA_MATRIX)
        
        (horizon_angle, vertical_angle) = get_angles_gerneal(image_points, rotation_vector, translation_vector, CAMERA_MATRIX, imgtest) 
        
	print("----------------")
        print(horizon_angle, vertical_angle)
        print("----------------")
        
        
        
        
        for p in image_points:
            cv2.circle(imgtest, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
        plt.imshow(face_detect)
        plt.show()

