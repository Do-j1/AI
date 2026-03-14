import cv2
import numpy as np
import mediapipe as mp

# Most of the code taken from the mediapipe code example 

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_image(rgb_image, detection_result, letter):

    annotated_image = np.array(rgb_image, dtype=np.uint8)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # ===== Add extra space on the left =====
    extra_space = 200  # pixels for text
    height, width, _ = annotated_image.shape

    bigger_image = np.zeros((height, width + extra_space, 3), dtype=np.uint8)
    bigger_image[:, extra_space:] = annotated_image

    hand_landmarks_list = detection_result.hand_landmarks

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # draw landmarks shifted to the right
        mp_drawing.draw_landmarks(
            bigger_image[:, extra_space:],  # draw only on the original image part
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    # ===== Draw letter text on the left =====
    cv2.putText(
        bigger_image,
        f"{letter}",
        (20, height // 2),  # position in the left margin
        cv2.FONT_HERSHEY_DUPLEX,
        2,  # bigger font
        HANDEDNESS_TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    return bigger_image

