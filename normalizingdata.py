import numpy as np

#Important: have to do this step when taking input from webcam
def normalizingdata(landmarks):
    landmarks = np.array(landmarks)

    # translate so wrist is origin
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # normalize hand size
    distances = np.linalg.norm(landmarks, axis=1)
    max_distance = np.max(distances)

    if max_distance > 0:
        landmarks = landmarks / max_distance

    return landmarks