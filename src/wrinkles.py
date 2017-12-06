import numpy as np
import cv2
import matplotlib.pyplot as plt
from sobel import SB


# Function used to calculate wrinkle density as per instructions
def wrinkle_density(img, threshold):
    Wa = np.sum(img >= threshold)
    Pa = img.shape[0] * img.shape[1]
    result = Wa/Pa
    return result

# Function used to calculate wrinkle depth as per instructions
def wrinkle_depth(img, threshold):
    Wa = img[img >= threshold]
    M = np.sum(Wa)
    result = M / (255*len(Wa))
    return result


# Function used to calculate average skin variance as per instructions
def avg_skin_variance(img):
    M = np.sum(img)
    Pa = img.shape[0] * img.shape[1]
    result = M / (255*Pa) 
    return result


# Function used to calculate the wrinkle features for the 5 different parts of the face: forehead, left eye, right eye, left cheek, right cheek
def wrinkle_features(img, threshold, eyes, left_eye, right_eye, mouth): 
    corner_left_eye = {}
    corner_right_eye = {}
    forehead = {}
    cheek_left = {}
    cheek_right = {}

    #Apply sobel to get wrinkled image
    sb = SB()
    wrinkled = sb.sobel(img)


    #Define the sections for each part of the face and calculate their wringle features 
    left_patch = left_eye[0] - max(left_eye[2], right_eye[2])//4
    right_patch = left_eye[0]
    top_patch = left_eye[1]
    bottom_patch = left_eye[1] + left_eye[3]
    window = wrinkled[top_patch:bottom_patch, left_patch:right_patch]
    corner_left_eye['density'] = wrinkle_density(window, threshold)
    corner_left_eye['depth'] = wrinkle_depth(window, threshold)
    corner_left_eye['variance'] = avg_skin_variance(window)


    left_patch = right_eye[0] + right_eye[2]
    right_patch = left_patch + max(left_eye[2], right_eye[2])//4
    top_patch = right_eye[1]
    bottom_patch = right_eye[1] + right_eye[3]
    window = wrinkled[top_patch:bottom_patch, left_patch:right_patch]
    corner_right_eye['density'] = wrinkle_density(window, threshold)
    corner_right_eye['depth'] = wrinkle_depth(window, threshold)
    corner_right_eye['variance'] = avg_skin_variance(window)

    left_for = mouth[0]
    right_for = mouth[0] + mouth[2]
    top_for = bottom_for - max(mouth[0]-left_eye[0], 
            right_eye[1]+right_eye[2]-mouth[0]-mouth[2])
    bottom_for = eyes - max(left_eye[3], right_eye[3]) 
    window = wrinkled[top_for:bottom_for, left_for:right_for]
    forehead['density'] = wrinkle_density(window, threshold)
    forehead['depth'] = wrinkle_depth(window, threshold)
    forehead['variance'] = avg_skin_variance(window)


    left_cheek_left = left_eye[0]
    left_cheek_right = left_cheek_left + (mouth[0] - left_eye[0])
    left_cheek_top = eyes + min(left_eye[3], right_eye[3])//4
    left_cheek_bottom = mouth[1]
    window = wrinkled[left_cheek_top:left_cheek_bottom, left_cheek_left:left_cheek_right]
    cheek_left['density'] = wrinkle_density(window, threshold)
    cheek_left['depth'] = wrinkle_depth(window, threshold)
    cheek_left['variance'] = avg_skin_variance(window)


    right_cheek_left = rb - (right_eye[0]+right_eye[2]-mouth[0]-mouth[2])
    right_cheek_right = right_eye[0] + right_eye[2]
    right_cheek_top = eyes + min(left_eye[3], right_eye[3])//4
    right_cheek_bottom = mouth[1]
    window = wrinkled[right_cheek_top:right_cheek_bottom, right_cheek_left:right_cheek_right]
    cheek_right['density'] = wrinkle_density(window, threshold)
    cheek_right['depth'] = wrinkle_depth(window, threshold)
    cheek_right['variance'] = avg_skin_variance(window)

    return corner_left_eye, corner_right_eye, forehead, cheek_left, cheek_right
