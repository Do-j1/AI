import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from draw_hand_landmarks import draw_landmarks_on_image
from normalizingdata import normalizingdata

#Settings for displaying the images
SHOW_IMAGES = False
NUM_IMAGES_SHOWN = 200
DELAY = 330

# List of letters to process
letters = ["aleff", "bb", "taa", "thaa", "jeem", "haa", "khaa", 
           "dal", "thal", "ra", "zay", "seen", "sheen", "saad", 
           "dhad", "ta", "dha", "ain", "ghain", "fa", "gaaf", 
           "kaaf", "laam", "meem", "nun", "ha", "waw", "ya",
            "la", "al", "toot", "yaa"]

# Base folder containing folders for each letter
base_folder = "hand_images"  # e.g., "hand_images/aleff/aleff (1).jpg"

# Create MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)


# Dictionary to store all landmarks
all_letters_landmarks = {}

#Going through every letter
for letter in letters:

    #Opening each letters folder
    letter_folder = os.path.join(base_folder, letter)
    if not os.path.exists(letter_folder):
        print(f"Folder not found: {letter_folder}")
        continue

    letter_landmarks = []

    # Loop through images in the folder
    for i, file_name in enumerate(sorted(os.listdir(letter_folder))):
        
        #limit the number of images displayed - computed if needed (testing)
        if i >= NUM_IMAGES_SHOWN and SHOW_IMAGES:
          break
        
        if not file_name.lower().endswith((".jpg", ".png")):
            continue

        # Reading each image to cv2
        image_path = os.path.join(letter_folder, file_name)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None: 
            print(f"Failed to load image: {image_path}")
            continue

        # RGBA → RGB if needed
        if img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Giving the image to mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)

        # If landmarks detected then store them in the dicitionary
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                hand_points = normalizingdata(hand_points) #normalizing data
                letter_landmarks.append(hand_points)


            # Optional: display images 
            if SHOW_IMAGES:
              annotated_image = draw_landmarks_on_image(img_rgb, detection_result,letter )
              cv2.imshow(f"{letter} landmarks", annotated_image)
              cv2.waitKey(DELAY)
              cv2.destroyAllWindows()


    #For each letter create a sepearte array
    all_letters_landmarks[letter] = np.array(letter_landmarks)
    print(f"{letter} landmarks shape: {all_letters_landmarks[letter].shape}") #See the Matrix shape for each letter

# Saving the data to disk
np.save("all_letters_landmarks.npy", all_letters_landmarks)
print("Saved all letters landmarks to all_letters_landmarks.npy")

