## Requirements

- **Python:** 3.13  
- **Libraries:**
  - `numpy` 2.4.3
  - `matplotlib` 3.10.8
  - `scikit-learn` 1.8.0
  - `mediapipe` 0.10.32
  - `opencv-python` 4.13.0.92

## Dataset Reference

Boulesnane, A., Bellil, L., & Ghiri, M. G. (2024). *ASLAD-190K: Arabic Sign Language Alphabet Dataset consisting of 190,000 images* (Version 1).  

- **Hand landmarks data:** saved in `all_letters_landmarks.npy`  
- **Format:** NumPy array (`numpy.save`)  
- **Shape:** `(133,522, 21, 3)` → 133,522 images, each with 21 hand landmarks (x, y, z)
- **Format:** dictionary {letter_name: list of images}

letters = ["aleff", "bb", "taa", "thaa", "jeem", "haa", "khaa", 
           "dal", "thal", "ra", "zay", "seen", "sheen", "saad", 
           "dhad", "ta", "dha", "ain", "ghain", "fa", "gaaf", 
           "kaaf", "laam", "meem", "nun", "ha", "waw", "ya",
            "la", "al", "toot", "yaa"]

## Dataset Processing

The original images have been translated into hand landmarks using MediaPipe.
Each hand contains 21 landmarks, color-coded for visualization:

**Wrist:** landmark 0
**Fingers:** landmarks 1–20, each with distinct coloring to differentiate fingers and joints

**Normalization**:

- The wrist (landmark 0) is used as the center point.
- All other landmarks are scaled relative to the wrist.

This ensures consistency regardless of hand size or position.

For details, see normalizing_data.py.


## Important when getting the hand landmarks from the webcam make sure to translate and normialze the data the same way in the file normalizing_data.py