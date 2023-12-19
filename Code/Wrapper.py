import os
import time
import sys
import csv

import cv2
import numpy as np
import argparse
import glob
import torch
from PIL import Image
import matplotlib.pyplot as plt
from threading import Thread
from djitellopy import Tello

from ultralytics import YOLO #YOLO

sys.path.append('RAFT/core')  # This is in raft folder
from raft import RAFT  # in RAFT folder
from utils import flow_viz  # in RAFT folder
from utils.utils import InputPadder  # in RAFT folder

DEVICE = 'cuda'
current_path = os.path.abspath(__file__)
current_path = current_path.replace("Wrapper.py", "")
####Optical flow for unknown gap related functions##################
def load_image(imfile):
    img = torch.from_numpy(imfile).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def contour_area(contour):
    return cv2.contourArea(contour)

def postprocess(i, current_path, image_path):
    # Load the image
    image = cv2.imread(image_path)
    # If the image has an alpha channel, remove it
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    real_path = current_path + f"/flow_frames/frame{2*i+1:03d}.png"
    real_image = cv2.imread(real_path)
    # real_image2 = cv2.imread(real_path)
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 1: Noise reduction with Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    # Use adaptive thresholding to get a binary image
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 2)
    # Use Canny edge detection
    edges = cv2.Canny(adaptive_thresh, 50, 150)
    # Dilate the edges to close the gaps
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    print("statement 4")
    # Apply closing to fill in gaps
    closed_edges = cv2.morphologyEx(
        dilated_edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Step 4: Find contours of the remaining objects (gaps)
    contours, hierarchy = cv2.findContours(
        closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    newlist_contours = contours
    newlist_contours = sorted(
        newlist_contours, key=cv2.contourArea, reverse=True)[:1]
    max_contour = newlist_contours[0]
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    # Sort contours by area and get the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    hulls = sorted(hull_list, key=cv2.contourArea, reverse=True)[:1]

    # Draw the largest contour and centroid if it exists
    for contour in contours:
        # Threshold to filter small contours (tunable)
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
            cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 3)
            cv2.drawContours(image, hulls, -1, (0, 0, 255), 3)
            cv2.drawContours(real_image, [contour], -1, (0, 255, 0), 3)
            cv2.drawContours(real_image, [max_contour], -1, (0, 255, 0), 3)
            cv2.drawContours(real_image, hulls, -1, (0, 0, 255), 3)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                cv2.circle(real_image, (cX, cY), 7, (255, 255, 255), -1)
                print("The centroid of the largest contour detected is:", cX, ",", cY)

    filepath = current_path + f"/flow_center/frame{i:03d}.png"
    cv2.imwrite(filepath, image)
    filepath2 = current_path + f"/flow_center_real/frame{i:03d}.png"
    cv2.imwrite(filepath2, real_image)
    # filepath3 = current_path + f"/contour_real/frame{i:03d}.png"
    # cv2.imwrite(filepath3, real_image2)

    return cX, cY

######### YOLO Related functions for phase 1 #######################################

# ----------pnp visualizations-----------------------------------


def draw_axis(img, rotvec, tvec, K): # Draw the 3d axis at the window center after pnp
    # unit is mm
    dist_coeffs = np.zeros((5, 1))
    dist_coeffs[0][0] = 0.02456386593401987
    dist_coeffs[1][0] = -0.5958069654037562
    dist_coeffs[2][0] = -0.0003932676388405013
    dist_coeffs[3][0] = -0.00017064279541258975
    dist_coeffs[4][0] = 1.8486532081847153
    points = np.float32(
        [[20, 0, 0], [0, 20, 0], [0, 0, 20], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotvec, tvec, K, dist_coeffs)
    axisPoints = axisPoints.astype(int)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[0].ravel()), (255, 0, 0), 3)  # Blue is x axis
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[1].ravel()), (0, 255, 0), 3)  # Green is Y axis
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
        axisPoints[2].ravel()), (0, 0, 255), 3)  # Red is z axis
    return img


# ------Corner infer funcs--------------------------------------
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def get_corners(img, mask,i): # Get the corners from window mask
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((3, 3), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is convolved and third parameter is the number
    # iterations will determine how much you want to erode/dilate a given image.
    img_erosion = cv2.erode(mask, kernel, iterations=15)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=15)

    img_resized = resize_with_aspect_ratio(img, width=960)
    resized_image = resize_with_aspect_ratio(
        mask, width=960)  # You can adjust the width as needed
    resized_image_erosion = resize_with_aspect_ratio(
        img_erosion, width=960)  # You can adjust the width as needed
    resized_image_dilation = resize_with_aspect_ratio(
        img_dilation, width=960)  # You can adjust the width as needed

    contours, _ = cv2.findContours(
        np.uint8(resized_image_dilation), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    resized_image_dilation_color = cv2.cvtColor(
        resized_image_dilation, cv2.COLOR_GRAY2BGR)

    corners = []
    for contour in contours:
        # Approximate polygon and ensure it has 4 corners
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Draw circles on the corner points
            for point in approx:
                x, y = point[0]
                cv2.circle(img_resized, (int(x), int(y)), 7, (0, 0, 255), -1)
                corners.append((x, y))

    # cv2.imshow('Input', resized_image)
    # cv2.imshow('Erosion', resized_image_erosion)
    # cv2.imshow('Dilation', resized_image_dilation_color)
    # cv2.imshow('Detected contours', img)

    corner_filename = current_path + str("/yolo_corners/") + f"frame{i:03d}.png"
    cv2.imwrite(corner_filename, img_resized)

    return corners, img_resized


def order_points(points):
    # Sort points by x-coordinate (leftmost will be top-left, rightmost will be top-right)
    sorted_points = sorted(points, key=lambda x: x[0])
    print("sorted func output", sorted_points)

    # left most and right most points
    left1 = sorted_points[0]
    left2 = sorted_points[1]

    right1 = sorted_points[-1]
    right2 = sorted_points[-2]

    if left1[1] > left2[1]:
        bottom_left = left1
        top_left = left2
    else:
        bottom_left = left2
        top_left = left1

    if right1[1] > right2[1]:
        top_right = right2
        bottom_right = right1
    else:
        top_right = right1
        bottom_right = right2

    return [top_left, bottom_left, bottom_right, top_right]


def get_axis(img, corners,i): # Solve the pnp for phase 1 windows
    K = np.array([[917.3527180617801, 0.0, 480.97134568905716], [
                 0.0, 917.1043451426654, 365.57078783755276], [0.0, 0.0, 1.0]])
    points_2D = np.array([corners], dtype="double")
    points_3D = np.array([
                        (-50.8, 45.72, 0),     # First
                        (-50.8, -45.72, 0),  # Second
                        (50.8, -45.72, 0),  # Third
                        (50.8, 45.72, 0)  # Fourth
    ])
    dist_coeffs = np.zeros((5, 1))
    dist_coeffs[0][0] = 0.02456386593401987
    dist_coeffs[1][0] = -0.5958069654037562
    dist_coeffs[2][0] = -0.0003932676388405013
    dist_coeffs[3][0] = -0.00017064279541258975
    dist_coeffs[4][0] = 1.8486532081847153
    success, rotation_vector, translation_vector = cv2.solvePnP(
        points_3D, points_2D, K, dist_coeffs, flags=0)
    image = draw_axis(img, rotation_vector, translation_vector, K)
    axis_filename = current_path + str("yolo_axes/") + f"frame{i:03d}.png"
    cv2.imwrite(axis_filename, image)
    return rotation_vector, translation_vector

#--------------------Dynamic window functions-------------------
def get_dynamic_inference(frame_path,i): # This function reads camera frames and returns the blue window outer corners and the angle made by hand w.r.t 12 o clock(measured in counter clock wise direction with 0 degrees at 12:00 clock)
    # You need to provide the correct path where the image is stored
    img = cv2.imread(frame_path)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ####################################################################################################################################################
    # Define the range for pink color
    lower_pink = np.array([150, 100, 100]) # Lower end of HSV range for pink at night
    upper_pink = np.array([200, 255, 255]) # Upper end of HSV range for pink at night
    # # Define the range for dark blue color (for the square) Worked yesterday
    # lower_blue = np.array([70, 50, 100])
    # upper_blue = np.array([130, 255, 255])
    # Define the range for dark blue color (for the square)
    lower_blue = np.array([70, 50, 40])
    upper_blue = np.array([130, 255, 255])
    ###################################################################################################################################################
    # Threshold the HSV image to get only dark blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Threshold the HSV image to get only pink colors
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Apply morphological operations for clock hand
    kernel_dilate = np.ones((3,3), np.uint8)
    kernel_erode = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_dilate)
    # Dilate to restore lost parts
    dilated = cv2.dilate(mask, kernel_dilate, iterations=1)
    # Erode to remove noise
    eroded = cv2.erode(dilated, kernel_erode, iterations=1)

    # Apply morphological operations for blue square
    kernel_blue_dilate = np.ones((3,3), np.uint8)
    kernel_blue_erode = np.ones((7,7), np.uint8)
    dilated_blue = cv2.dilate(blue_mask, kernel_blue_dilate, iterations=3)
    eroded_blue = cv2.erode(dilated_blue, kernel_blue_erode, iterations=2)

    added_image = eroded + eroded_blue
    # Find contours for the hand and the square window
    contours_for_square, _ = cv2.findContours(added_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    square_contour = max(contours_for_square, key=cv2.contourArea)

    # Calculate the centroid of the square contour for the 12:00 orientation
    M = cv2.moments(square_contour)
    if M["m00"] != 0:
        square_cX = int(M["m10"] / M["m00"])
        square_cY = int(M["m01"] / M["m00"])
    else:
        square_cX, square_cY = 0, 0
    orientation_12 = (square_cX, square_cY)

    # Draw the vertical line passing through the centroid of the square
    added_colored_image = cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR)
    cv2.line(added_colored_image, (square_cX, square_cY), (square_cX, square_cY - 100), (255, 255, 0), 2)


    # Centroid of the square
    centroid = (square_cX, square_cY)

    # Point directly above the centroid, 100 pixels up
    above_centroid = (square_cX, square_cY - 100)

    # The vector for the line is the difference between the two points
    # In this case, it would be:
    line_vector = (above_centroid[0] - centroid[0], above_centroid[1] - centroid[1])


    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the rotating hand
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the extreme points
    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

    # print(centroid,topmost,bottommost)


    # Calculate the distance from the centroid to the topmost and bottommost points
    distance_to_topmost = np.sqrt((topmost[0] - centroid[0])**2 + (topmost[1] - centroid[1])**2)
    distance_to_bottommost = np.sqrt((bottommost[0] - centroid[0])**2 + (bottommost[1] - centroid[1])**2)

    # Depending on which distance is greater, use the corresponding point to form the hand vector
    if distance_to_topmost > distance_to_bottommost:
        # If the hand is in the upper half of the window
        hand_vector = (topmost[0] - centroid[0], topmost[1] - centroid[1])
    else:
        # If the hand is in the lower half of the window
        hand_vector = (bottommost[0] - centroid[0], bottommost[1] - centroid[1])

    # Normalize the vectors to get their unit vectors
    line_unit_vector = line_vector / np.linalg.norm(line_vector)
    hand_unit_vector = hand_vector / np.linalg.norm(hand_vector)

    # Calculate the angle between the two vectors using the dot product
    dot_product = np.dot(line_unit_vector, hand_unit_vector)
    angle_rad = np.arccos(dot_product)

    # Since np.arccos gives the angle in radians, convert it to degrees
    angle_deg = np.degrees(angle_rad)


    # Ensure the angle is between 0 and 360 degrees
    if hand_vector[0] < 0:
        angle_from_12 = angle_deg
    else:
        angle_from_12 = 360 - angle_deg

    # Display the results
    cv2.putText(added_colored_image, "Angle: {:.4f}".format(angle_from_12), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    print(f"The angle of the rotating hand from the 12:00 position is: {angle_from_12} degrees")
    ##resize the eroded contours image of the (optional)
    # eroded_blue_resized = resize_with_aspect_ratio(eroded_blue, width=900) # You can adjust the width as needed
    eroded_blue_resized = eroded_blue
    # Find contours for the eroded blue mask
    blue_contours, _ = cv2.findContours(eroded_blue_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_blue_contours = sorted(blue_contours, key=cv2.contourArea, reverse=True)
    eroded_blue_resized_color = cv2.cvtColor(eroded_blue_resized, cv2.COLOR_GRAY2BGR)
    corners = []
    for blue_contour in sorted_blue_contours:
        epsilon = 0.05 * cv2.arcLength(blue_contour, True)
        approx = cv2.approxPolyDP(blue_contour, epsilon, True)

        print(f"Found {len(approx)} corners")  # Debugging line

        if len(approx) == 4:
            for point in approx:
                x, y = point[0]
                print("x is and y is ",x,y)
                cv2.circle(added_colored_image, (int(x), int(y)), 7, (0, 0, 255), -1)
                corners.append((x, y))
            break
        else:
            print("Not a quadrilateral")  # Debugging line
    filepath = current_path + f"dynamic_result/frame{i:03d}.png"
    cv2.imwrite(filepath, added_colored_image)
    return angle_from_12,corners


def get_axis_dynamic(img, corners,i): # PNP solve for phase 3
    K = np.array([[917.3527180617801, 0.0, 480.97134568905716], [
                 0.0, 917.1043451426654, 365.57078783755276], [0.0, 0.0, 1.0]])
    points_2D = np.array([corners], dtype="double")
    points_3D = np.array([
                        (-42.0, 42.0, 0),     # First
                        (-42.0, -42.0, 0),  # Second
                        (42.0, -42.0, 0),  # Third
                        (42.0, 42.0, 0)  # Fourth
    ])
    dist_coeffs = np.zeros((5, 1))
    dist_coeffs[0][0] = 0.02456386593401987
    dist_coeffs[1][0] = -0.5958069654037562
    dist_coeffs[2][0] = -0.0003932676388405013
    dist_coeffs[3][0] = -0.00017064279541258975
    dist_coeffs[4][0] = 1.8486532081847153
    success, rotation_vector, translation_vector = cv2.solvePnP(
        points_3D, points_2D, K, dist_coeffs, flags=0)
    image = draw_axis(img, rotation_vector, translation_vector, K)
    axis_filename = current_path + str("/dynamic_axes/") + f"frame{i:03d}.png"
    cv2.imwrite(axis_filename, image)
    return rotation_vector, translation_vector

#-----------Main code------------------------------------------------------
def main():
    try:

        #--------Create folders-----------------------------------------------
        folders_list = ["flow_center", "flow", "flow_outputs", "flow_frames",
                        "frames_thread", "flow_center_real", "flow_contour_real",
                        "yolo_frames","yolo_corners","yolo_masks","yolo_axes",
                        "dynamic_result","dynamic_frame","dynamic_axes"]
        for folder in folders_list:
            folder_path = os.path.join(current_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        #-----------YOLO load------------------------------------------------
        yolo_model_path = current_path + \
            str("YOLO Model/runs/segment/train2/weights/last.pt")

        yolo_model = YOLO(yolo_model_path)
        #--------RAFT load---------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision',
                            action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true',
                            help='use efficent correlation implementation')
        args = parser.parse_args()
        raft_model = torch.nn.DataParallel(RAFT(args))
        raft_model.load_state_dict(torch.load(args.model))

        raft_model = raft_model.module
        raft_model.to(DEVICE)
        raft_model.eval()

        #----------------------------------------------------------
        drone = Tello()
        drone.connect()
        # CHECK SENSOR READINGS------------------------------------------------------------------
        print('Altitude ', drone.get_distance_tof())
        print('Battery, ', drone.get_battery())
        print('Temperature, ', drone.get_temperature())

        drone.streamon()
        #--------Ignore black frames------------------
        for j in range(0, 7):  # Ignores black frames
            frame1 = drone.get_frame_read().frame
        # # Thread(target=recordWorker).start()
        drone.takeoff()
        time.sleep(2)


        #-----------RAFT parameters-------
        image_center = np.array([480, 360])
        raft_tol = 200  # 200, 250
        runs = 0
        framei = 0
        flowi = 0
        centers_dict = {}

        #---------PHASE 1:YOLO-------------------------------------
        window_count = 0 # YOLO parameter
        window_tol = 15 # YOLO parameter
        yolo_i = 0
        ##########--WAYPOINTS--######################################################################################
        # NOTE: x - depth, y - +ve: left side, -ve: right side, z - +ve: Up, yaw - clockwise: face right, CCW: face left
        # WINDOW 1 RELATED--------------------------------------------------------------------------------------------
        wp1 = [50,95,110] # This is the first point you go to after takeoff.(Make sure full RED window is visible)(NO X )
        speed1 = 95 # Speed with which we go from takeoff to waypoint 1
        wp2 = [285, 0, -50] #ONCE ALIGNED WITH WINDOW 1. GO INTO IT. HERE WE DECIDE DEPTH INTO WINDOW AND HEIGHT REDUCTION.(NO Y)
        speed2 = 85 # Speed with which we go into the WINDOW 1.
        # CROSSED WINDOW 1; WINDOW 2 RELATED------------------------------------------------------------------------------------------
        yaw_new = 45
        wp3 = [0,0,60] # This is the point we go to view whole WINDOW 2. (NO X)
        speed3 = 95 # Speed with which we go to waypoint 3

        wp4 = [220,0,-60] # AFTER CORRECTION use this point to go inside window 2. (NO Y)
        speed4 = 65 # Corresponding speed of wp4
        # CROSSED WINDOW 2; UNKNOWN GAP RELATED--------------------------------------------------------------------------------------------
        #-----Transition to unknown gap-------------
        # Ideally we want the drone to (in order of priority): 1. reduce the height to align with gap center,2. Change yaw to face the gap directly
        # 3. align the y center directly with gap within tolerance so no further correction is needed. 4. drone should be able to see whole gap
        # Here we need to control: 1. Reduce Z; 2. Change YAW; 3. Change Y to align with center; 4. Add +\- X for visibility of gap.
        wp5 = [0,75,-45] # This reduces the height to gap height. Aligns y direction with center.
        wp_new = [85,0,0]
        speed5 = 95# speed of wp5
        yaw5 = 80 # Amount of rotation in degrees
        x_correction5 = -60 # Move back to get full view
        #-------------------------------------------
        # DETECTED AND ALIGNED WITH GAP AFTER FLOW ESTIMATE; GO THROUGH THE UNKNOWN GAP
        wp6 = [270, 0, 0] # HOW FAR to go inside the gap
        speed6 = 95
        # DRONE CROSSED GAP; TRANSITION TO DYNAMIC WINDOW
        wp7 = [0,20,100] # THIS adjusts y and height for dynamic window to be fully visible
        speed7 = 95
        yaw7 = 100
        x_correction7 = 50
        # y_correction7 =
        # GOT THE CENTER TO GO ON DYNAMIC WINDOW
        z_reduction8 =70 # HOW MUCH HEIGHT TO SUBTRACT "ADDITIONALLY" FROM CENTER TO GO THROUGH the hole(INPUT POSITIVE VALUE AS WE SUBTRACT BELOW)
        x_depth8 = 50# HOW MUCH "ADDITIONAL" DEPTH TO GO INSIDE THE WINDOW
        speed8 = 95 # HOW FAST TO GO INSIDE (IMPORTANT HERE) 65 prev
        #################################################################################################################################################
        # TUNING PARAMETERS: COLOR THRESHOLDS, ANGLE(DYNAMIC WINDOW),GAP TOLERANCE(in pixel), RED WINDOW CORRECTION TOLERENCE(IN cm), SLEEP TIMES,SPEEDS, POSTPROCESSING ITERATIONS
        # python3 Wrapper.py --model=RAFT/models/raft-sintel.pth
        # kill -9 $(ps -A | grep python | awk '{print $1}')


        while True:
            print("Running phase 1:")
            while window_count < 2:
                if window_count ==0:
                    drone.go_xyz_speed(wp1[0],wp1[1],wp1[2],speed1) # WAYPOINT 1############################################################
                elif window_count == 1:
                    drone.rotate_clockwise(yaw_new)
                    drone.go_xyz_speed(wp3[0],wp3[1],wp3[2],speed3) # WAYPOINT 3#############################################################

                while True:
                    yolo_frame = drone.get_frame_read().frame
                    yolo_frame = cv2.cvtColor(yolo_frame, cv2.COLOR_RGB2BGR)
                    H, W, _ = yolo_frame.shape
                    print("yolo shape", (H, W))
                    filename = current_path + str("/yolo_frames/") + f"frame{yolo_i:03d}.png"
                    cv2.imwrite(filename, yolo_frame)

                    yolo_frame = cv2.imread(filename)
                    yolo_results = yolo_model(yolo_frame)
                    try:
                        yolo_mask = yolo_results[0].masks.data
                        yolo_mask = yolo_mask.cpu().numpy()*255
                        yolo_mask = cv2.resize(yolo_mask[0], (W, H))
                        mask_filename = current_path + \
                            str("/yolo_masks/") + f"frame{yolo_i:03d}.png"
                        cv2.imwrite(mask_filename, yolo_mask)

                        # To get corners
                        yolo_corners, yolo_corner_img = get_corners(yolo_frame, yolo_mask,yolo_i)
                        # Reorder the corners
                        yolo_corners = order_points(yolo_corners)
                        rvec, tvec = get_axis(yolo_corner_img, yolo_corners,yolo_i)
                        print("tvec 0",tvec[0][0],"tolerance",window_tol)
                        print("tvec 1",tvec[1][0],"tolerance",window_tol)
                        print("tvec 2",tvec[2][0],"tolerance",window_tol)
                        if abs(tvec[0][0]) < window_tol:
                            # drone.go_xyz_speed(int((1.5*tvec[2][0]+0.3*groundtruth_list[counter][0])/2), int(
                            #     (-tvec[0][0] + 0)/2), int((-tvec[1][0]+0)/2)-10, speed)
                            print("Within tolerance")
                            if window_count ==0:
                                drone.go_xyz_speed(wp2[0], wp2[1], wp2[2], speed2) # WAYPOINT 2###########################################################
                            elif window_count ==1:
                                drone.go_xyz_speed(wp4[0],wp4[1],wp4[2],speed4) # WAYPOINT 4###############################################################
                            yolo_i += 1
                            break
                        else:
                            print("Need correction")
                            drone.go_xyz_speed(0, -int(tvec[0][0]), 0, 45) # CORRECTION in Y direction (Add logic for out of range)
                            # Correction window center and image center
                            yolo_i += 1

                    except Exception as error:
                        print(f"An error occurred at yolo:{type(error).__name__} - {error}")
                        continue
                window_count += 1
            print("Done with phase 1. Phase 2 begins")
            #-----------Phase 2--------------------------------------------------------------------------------------------------------------------------------------
            ###########--UNKNOWN GAP--#####################################################################################################################
            drone.rotate_counter_clockwise(yaw5)  # THIS BIT CORRECTS YAW FOR DRONE TO LOOK DIRECTLY AT GAP
            time.sleep(0.5)
            drone.go_xyz_speed(wp_new[0],wp_new[1],wp_new[2],speed5) #THIS BIT CORRECTS HEIGHT (Z) and CORRECTS Y for CENTER ALIGNMENT # WAYPOINT 5
            time.sleep(0.5)
            drone.go_xyz_speed(wp5[0],wp5[1],wp5[2],speed5) #THIS BIT CORRECTS HEIGHT (Z) and CORRECTS Y for CENTER ALIGNMENT # WAYPOINT 5
            time.sleep(0.5)

            # drone.go_xyz_speed(x_correction5,0,0,speed5) #THIS BIT CORRECTS X FOR GAP TO BE FULLY VISIBLE
            # time.sleep(0.5)
            ###############################################################################################################################################
            print("Starting flow detection")
            while True:
                frame_no = 0
                center_list = []
                while frame_no < 1:

                    try:
                        # ------Servoing------------------
                        drone.move_up(20)
                        time.sleep(2)
                        drone.move_down(20)
                        time.sleep(1.3)
                        # ----Reading two frames---------
                        print("Reading frame 1")
                        frame1 = drone.get_frame_read().frame
                        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
                        H, W, _ = frame1.shape
                        print("flow shape", (H, W))
                        filename = current_path + \
                            str("/flow_frames/") + f"frame{framei:03d}.png"
                        cv2.imwrite(filename, frame1)
                        frame1 = cv2.imread(filename)
                        framei += 1
                        time.sleep(0.3)  # keep it low before 0.2

                        frame2 = drone.get_frame_read().frame
                        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                        H, W, _ = frame2.shape
                        print("flow shape", (H, W))
                        filename = current_path + \
                            str("/flow_frames/") + f"frame{framei:03d}.png"
                        cv2.imwrite(filename, frame2)
                        frame2 = cv2.imread(filename)
                        # -----------------------------------------
                        # -------Get optical flow and do post-processing------------------
                        with torch.no_grad():
                            print("starting nn")
                            image1 = load_image(frame1)
                            image2 = load_image(frame2)
                            print("loaded images")
                            padder = InputPadder(image1.shape)
                            image1, image2 = padder.pad(image1, image2)
                            print("padding done")
                            flow_low, flow_up = raft_model(
                                image1, image2, iters=20, test_mode=True)
                            print("flow model ran")

                            raft_img = image1[0].permute(1, 2, 0).cpu().numpy()
                            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

                            # map flow to rgb image
                            flo = flow_viz.flow_to_image(flo)
                            img_flo = np.concatenate([raft_img, flo], axis=0)
                            image_path = current_path + \
                                str("/flow/")+f"frame{flowi:03d}.png"
                            cv2.imwrite(image_path, flo)
                            print("saved the flow", flowi)
                            # drone.send_keepalive()

                            cX, cY = postprocess(flowi, current_path, image_path)
                            center_list.append([cX, cY])
                            frame_no += 1
                            flowi += 1
                            framei += 1
                    except Exception as error:
                        print(f"An error occurred at raft:{type(error).__name__} - {error}")
                        continue
                runs += 1
                centers_dict[f"run{runs}"] = center_list

                # Find average center
                center_list = np.array(center_list)
                average_center = np.mean(center_list, axis=0)

                # -----Visual servoing algo------------------
                if np.linalg.norm(average_center-image_center) <= raft_tol:
                    drone.go_xyz_speed(wp6[0], wp6[1], wp6[2], speed6) # Go through window depth #WAYPOINT 6 ##################################################################################
                    # time.sleep(3)
                    # drone.land()
                    print(centers_dict)
                    break
                conversion_factor = 0.20  # 0.20, 0.18, 0.15
                if image_center[0] - average_center[0] > 0:
                    y_command = int(conversion_factor *
                                    (abs(image_center[0] - average_center[0])))
                    if y_command < 10:
                        drone.go_xyz_speed(wp6[0], wp6[1], wp6[2], speed6) #WAYPOINT 6 ##################################################################################
                        # drone.land()
                        break
                    elif 10 < y_command < 20:
                        drone.go_xyz_speed(0, 20, 0, 45)

                    else:
                        drone.go_xyz_speed(0, y_command, 0, 45)
                else:
                    y_command = -int(conversion_factor *
                                    (abs(image_center[0] - average_center[0])))
                    if y_command > 10:
                        drone.go_xyz_speed(wp6[0], wp6[1], wp6[2], speed6) #WAYPOINT 6 ##################################################################################
                        # drone.land()
                        break
                    elif 10 < y_command < 20:
                        drone.go_xyz_speed(0, -20, 0, 45)
                    else:
                        drone.go_xyz_speed(0, y_command, 0, 45)
                if image_center[1] - average_center[1] > 0:
                    z_command = int(conversion_factor *
                                    (abs(image_center[1] - average_center[1])))
                else:
                    z_command = -int(conversion_factor *
                                    (abs(image_center[1] - average_center[1])))
                print(
                    f"y command is: {int(conversion_factor*(abs(image_center[0] - average_center[0])))}")
                time.sleep(3) # CAN MODIFY
            print("Phase 2 ends here")
            #-------------------PHASE 3----------------------------------------------------------
            ######## Perform transition to DYNAMIC WINDOW-#############################################################################################################
            drone.go_xyz_speed(wp7[0],wp7[1],wp7[2],speed7) #THIS BIT CORRECTS HEIGHT (Z) and CORRECTS Y for CENTER ALIGNMENT
            time.sleep(0.5)
            drone.rotate_clockwise(yaw7) # THIS BIT CORRECTS YAW FOR DRONE TO LOOK DIRECTLY AT DYNAMIC WINDOW
            time.sleep(0.5)
            drone.go_xyz_speed(x_correction7,0,0,speed7) #THIS BIT CORRECTS X FOR DYNAMIC WINDOW TO BE FULLY VISIBLE
            time.sleep(0.5)
            #####################################################################################################################################################
            dyn_counter = 0
            while True:
                try:
                    dynamic_frame = drone.get_frame_read().frame
                    dynamic_frame = cv2.cvtColor(dynamic_frame, cv2.COLOR_RGB2BGR)
                    H, W, _ = dynamic_frame.shape
                    print("dynamic shape", (H, W))
                    filename_dynamic = current_path +f"dynamic_frame/frame{dyn_counter:03d}.png"
                    cv2.imwrite(filename_dynamic, dynamic_frame)
                    dynamic_frame = cv2.imread(filename_dynamic)
                    dynamic_angle,dynamic_corners = get_dynamic_inference(filename_dynamic,dyn_counter)
                    print("got corners",dynamic_corners)
                    dynamic_corners = order_points(dynamic_corners)
                    print("Ordered the points")
                    rvec, tvec = get_axis_dynamic(dynamic_frame, dynamic_corners,dyn_counter)
                    print("got tvec")
                    dyn_counter+=1
                    if dynamic_angle>170: # If the clock handle is greater than 170 degrees send drone to the 3D point below window center
                        # print("Go")
                        drone.go_xyz_speed(int(tvec[2][0])+x_depth8, -int((tvec[0][0])), -int(tvec[1][0])-z_reduction8, speed8) # WAYPOINT 8##################################################
                        drone.land()
                        break

                except Exception as error:
                    print(f"An error occurred at dynamic:{type(error).__name__} - {error}")
                    continue



    except KeyboardInterrupt:
        # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
        print('keyboard interrupt')
        # drone.emergency()
        drone.land()
        drone.emergency()
        drone.end()


if __name__=='__main__':
    main()
