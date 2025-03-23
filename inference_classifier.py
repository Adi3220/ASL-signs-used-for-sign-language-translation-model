# import pickle
#
# import cv2
# import mediapipe as mp
# import numpy as np
#
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
#
# cap = cv2.VideoCapture(0)
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 : 'R', 18 : 'S', 19 : 'T', 20 : 'U', 21 : 'V', 22 : 'W', 23 : 'X', 24 : 'Y', 25 : 'Z'}
# while True:
#
#     data_aux = []
#     x_ = []
#     y_ = []
#
#     ret, frame = cap.read()
#
#     H, W, _ = frame.shape
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#
#                 x_.append(x)
#                 y_.append(y)
#
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
#
#         prediction = model.predict([np.asarray(data_aux)])
#
#         predicted_character = labels_dict[int(prediction[0])]
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(100)
#
#
# cap.release()
# cv2.destroyAllWindows()

# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time
#
# # Load trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
#
# cap = cv2.VideoCapture(0)
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
#                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
#                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
#
# # Variables to store detected letters and words
# current_word = []
# final_sentence = ""
# last_predicted = None
# last_time = time.time()
#
# while True:
#     data_aux = []
#     x_ = []
#     y_ = []
#
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
#
#     predicted_character = None
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)
#
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
#
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = labels_dict[int(prediction[0])]
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
#
#     # Display the predicted character and formed words
#     cv2.putText(frame, f"Predicted: {predicted_character if predicted_character else '?'}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
#     cv2.putText(frame, f"Word: {''.join(current_word)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
#     cv2.putText(frame, f"Sentence: {final_sentence}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
#
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(100) & 0xFF
#
#     # Controls:
#     if key == ord('a') and predicted_character:  # Add letter to the word
#         current_word.append(predicted_character)
#         print(f"Added: {predicted_character}, Current Word: {''.join(current_word)}")
#
#     if key == ord('b'):  # Backspace: remove last letter
#         if current_word:
#             removed_letter = current_word.pop()
#             print(f"Removed: {removed_letter}, Current Word: {''.join(current_word)}")
#
#     if key == 13:  # Enter: Store final word in sentence
#         final_sentence += ''.join(current_word) + " "
#         current_word.clear()
#         print("Sentence so far:", final_sentence)
#
#     if key == ord(' '):  # Space: add space
#         final_sentence += " "
#         print("Sentence so far:", final_sentence)
#
#     if key == ord('q'):  # Quit
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # Final output
# print("\nFinal Output Sentence:", final_sentence)

#
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import pyttsx3  # Import text-to-speech library
#
# # Initialize text-to-speech engine
# engine = pyttsx3.init()
#
# # Load trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
#
# cap = cv2.VideoCapture(0)
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
#                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
#                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
#
# # Variables to store detected letters and words
# current_word = []
# final_sentence = ""
# last_predicted = None
# last_time = time.time()
#
# while True:
#     data_aux = []
#     x_ = []
#     y_ = []
#
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
#
#     predicted_character = None
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)
#
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
#
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = labels_dict[int(prediction[0])]
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
#
#     # Display the predicted character and formed words
#     cv2.putText(frame, f"Predicted: {predicted_character if predicted_character else '?'}", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
#     cv2.putText(frame, f"Word: {''.join(current_word)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
#     cv2.putText(frame, f"Sentence: {final_sentence}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
#
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(100) & 0xFF
#
#     # Controls:
#     if key == ord('a') and predicted_character:  # Add letter to the word
#         current_word.append(predicted_character)
#         print(f"Added: {predicted_character}, Current Word: {''.join(current_word)}")
#
#     if key == ord('b'):  # Backspace: remove last letter
#         if current_word:
#             removed_letter = current_word.pop()
#             print(f"Removed: {removed_letter}, Current Word: {''.join(current_word)}")
#
#     if key == 13:  # Enter: Store final word in sentence
#         final_sentence += ''.join(current_word) + " "
#         current_word.clear()
#         print("Sentence so far:", final_sentence)
#
#     if key == ord(' '):  # Space: add space
#         final_sentence += " "
#         print("Sentence so far:", final_sentence)
#
#     if key == ord('v'):  # Speak the current word or sentence
#         if current_word:
#             text_to_speak = ''.join(current_word)
#         else:
#             text_to_speak = final_sentence.strip()
#
#         print(f"Speaking: {text_to_speak}")
#         engine.say(text_to_speak)
#         engine.runAndWait()
#
#     if key == ord('q'):  # Quit
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # Final output
# print("\nFinal Output Sentence:", final_sentence)
#
# # Speak final sentence before exiting
# if final_sentence.strip():
#     print(f"Speaking Final Sentence: {final_sentence.strip()}")
#     engine.say(final_sentence.strip())
#     engine.runAndWait()


import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3  # Text-to-speech
import speech_recognition as sr  # Speech-to-text
import os  # For accessing images

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Variables to store detected letters and words
current_word = []
final_sentence = ""
spoken_text = ""  # Stores words converted from speech
last_predicted = None
last_time = time.time()

# Folder where sign language images are stored
sign_images_path = "sign_images/"

def recognize_speech():
    """Captures and converts speech to text"""
    global spoken_text
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            spoken_text = recognizer.recognize_google(audio).upper()
            print(f"🗣 Recognized: {spoken_text}")
        except sr.UnknownValueError:
            print("❌ Couldn't understand speech.")
            spoken_text = ""
        except sr.RequestError:
            print("⚠ Speech recognition service is unavailable.")
            spoken_text = ""

def show_sign_language(word):
    """Displays sign language images for the given word"""
    for letter in word:
        image_path = os.path.join(sign_images_path, f"{letter}.jpg")
        if os.path.exists(image_path):
            sign_img = cv2.imread(image_path)
            sign_img = cv2.resize(sign_img, (300, 300))
            cv2.imshow("Sign Language", sign_img)
            cv2.waitKey(1000)  # Show each letter for 1 second
        else:
            print(f"⚠ Image for '{letter}' not found!")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the predicted character and formed words
    cv2.putText(frame, f"Predicted: {predicted_character if predicted_character else '?'}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.putText(frame, f"Word: {''.join(current_word)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    cv2.putText(frame, f"Sentence: {final_sentence}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Spoken: {spoken_text}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(100) & 0xFF

    # Controls:
    if key == ord('a') and predicted_character:
        current_word.append(predicted_character)
        print(f"Added: {predicted_character}, Current Word: {''.join(current_word)}")

    if key == ord('b'):
        if current_word:
            removed_letter = current_word.pop()
            print(f"Removed: {removed_letter}, Current Word: {''.join(current_word)}")

    if key == 13:
        final_sentence += ''.join(current_word) + " "
        current_word.clear()
        print("Sentence so far:", final_sentence)

    if key == ord(' '):
        final_sentence += " "
        print("Sentence so far:", final_sentence)

    if key == ord('v'):
        text_to_speak = ''.join(current_word) if current_word else final_sentence.strip()
        print(f"🔊 Speaking: {text_to_speak}")
        engine.say(text_to_speak)
        engine.runAndWait()

    if key == ord('s'):
        recognize_speech()
        if spoken_text:
            show_sign_language(spoken_text)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
