import threading
import queue
import time
from gtts import gTTS
import pygame
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from draw_hand_landmarks import draw_landmarks_on_image
from normalizingdata import normalizingdata
from PIL import Image, ImageDraw, ImageFont

# إعدادات
SHOW_IMAGES = False
NUM_IMAGES_SHOWN = 200
THRESHOLD = 10
WORD_CLEAR_DELAY = 5  # الآن بعد 5 ثواني
LETTERS = ["aleff", "bb", "taa", "thaa", "jeem", "haa", "khaa",
           "dal", "thal", "ra", "zay", "seen", "sheen", "saad",
           "dhad", "ta", "dha", "ain", "ghain", "fa", "gaaf",
           "kaaf", "laam", "meem", "nun", "ha", "waw", "ya",
           "la", "al", "toot", "yaa"]

BASE_FOLDER = Path(r"C:\Users\saloo\AI-main\processing_data\hand_images")
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
LANDMARKS_FILE = "all_letters_landmarks.npy"
SPOKEN_FOLDER = Path(__file__).parent / "spoken_words"

SPOKEN_FOLDER.mkdir(exist_ok=True)

# التحقق من الملفات
if not BASE_FOLDER.exists():
    raise FileNotFoundError(f"Base folder not found: {BASE_FOLDER}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# إنشاء detector
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH.resolve())),
    num_hands=1,
    min_hand_detection_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)

# تحميل أو حساب landmarks
if Path(LANDMARKS_FILE).exists():
    print("✅ Loading saved landmarks...")
    all_letters_landmarks = np.load(LANDMARKS_FILE, allow_pickle=True).item()
else:
    print("⚡ Calculating landmarks...")
    all_letters_landmarks = {}
    for letter in LETTERS:
        letter_folder = os.path.join(BASE_FOLDER, letter)
        if not os.path.exists(letter_folder):
            continue
        letter_landmarks = []
        for file_name in os.listdir(letter_folder):
            if not file_name.lower().endswith((".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(letter_folder, file_name))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = detector.detect(mp_image)
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    pts = normalizingdata(pts)
                    letter_landmarks.append(pts)
        all_letters_landmarks[letter] = np.array(letter_landmarks)
    np.save(LANDMARKS_FILE, all_letters_landmarks)

# إعداد الكاميرا
cap = cv2.VideoCapture(0)

word = ""
last_letter = ""
counter = 0
last_update_time = time.time()

arabic_map = {
    "aleff":"ا","bb":"ب","taa":"ت","thaa":"ث","jeem":"ج","haa":"ح","khaa":"خ",
    "dal":"د","thal":"ذ","ra":"ر","zay":"ز","seen":"س","sheen":"ش","saad":"ص",
    "dhad":"ض","ta":"ط","dha":"ظ","ain":"ع","ghain":"غ","fa":"ف","gaaf":"ق",
    "kaaf":"ك","laam":"ل","meem":"م","nun":"ن","ha":"ه","waw":"و","ya":"ي",
    "la":"لا","al":"أل","toot":"توت","yaa":"ياء"
}

# خط عربي
font = ImageFont.truetype("arial.ttf", 40)

# إعداد الصوت
pygame.mixer.init()
speech_queue = queue.Queue()
spoken_text = ""  # لتجنب تكرار النطق

def speech_worker():
    global spoken_text
    while True:
        text = speech_queue.get()
        if text is None:
            break
        # توليد اسم ملف جديد لكل كلمة
        filename = SPOKEN_FOLDER / f"{text}_{int(time.time())}.mp3"
        if text != spoken_text:
            tts = gTTS(text=text, lang='ar')
            tts.save(filename)
            pygame.mixer.music.load(str(filename))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            spoken_text = text
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

print("🎬 Start signing... (Press ESC to exit, S to save hand sample)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    predicted_letter = ""

    if result.hand_landmarks:
        pts = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
        pts = normalizingdata(pts)

        min_dist = float("inf")
        for letter, data in all_letters_landmarks.items():
            if len(data) == 0:
                continue
            avg = np.mean(data, axis=0)
            dist = np.linalg.norm(np.array(avg) - np.array(pts))
            if dist < min_dist:
                min_dist = dist
                predicted_letter = letter

        if predicted_letter == last_letter:
            counter += 1
        else:
            counter = 0

        last_letter = predicted_letter

        if counter == THRESHOLD:
            new_letter = arabic_map.get(predicted_letter, predicted_letter)
            if not word or word[-1] != new_letter:
                word += new_letter
                last_update_time = time.time()
            counter = 0

        # 🔥 تدريب يدوي: اضغط S لتسجيل الحركة
        if cv2.waitKey(1) & 0xFF == ord('s'):
            letter_folder = os.path.join(BASE_FOLDER, predicted_letter)
            os.makedirs(letter_folder, exist_ok=True)
            filename = f"{len(os.listdir(letter_folder)) + 1}.png"
            cv2.imwrite(os.path.join(letter_folder, filename), frame)
            if predicted_letter not in all_letters_landmarks:
                all_letters_landmarks[predicted_letter] = []
            all_letters_landmarks[predicted_letter].append(pts)
            np.save(LANDMARKS_FILE, all_letters_landmarks)
            print(f"✅ Saved new sample for {predicted_letter}")

    # 🔊 نطق الكلمة بعد تأخير WORD_CLEAR_DELAY
    if word and (time.time() - last_update_time) > WORD_CLEAR_DELAY:
        speech_queue.put(word)
        print("Speaking:", word)
        word = ""
        spoken_text = ""  # إعادة تهيئة النص المنطوق

    # عرض النص
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text((50, 60), f"Letter: {predicted_letter}", font=font, fill=(0,255,0))
    draw.text((50, 120), f"Word: {word}", font=font, fill=(255,0,0))
    frame = np.array(frame_pil)

    cv2.imshow("Sign Language AI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# تنظيف
speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()