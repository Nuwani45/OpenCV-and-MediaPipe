import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

ret, frame = cap.read()
h, w, c = frame.shape

canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0

colors = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (0, 0, 0)       # Black (pen)
]

eraser_color = (0, 0, 0)
color = (255, 0, 0)
brush_size = 5

def draw_palette(img):
    # colors
    for i, c in enumerate(colors):
        cv2.rectangle(img, (i * 80, 0), ((i + 1) * 80, 80), c, -1)

    # eraser
    cv2.rectangle(img, (len(colors) * 80, 0),
                  ((len(colors) + 1) * 80, 80), (255, 255, 255), -1)
    cv2.putText(img, "E", (len(colors) * 80 + 25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

 
    cv2.rectangle(img, ((len(colors) + 1) * 80, 0),
                  ((len(colors) + 2) * 80, 80), (200, 200, 200), -1)
    cv2.putText(img, "C", ((len(colors) + 1) * 80 + 25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = draw_palette(img)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

        
            if y < 80:
                index = x // 80

                if index < len(colors):
                    color = colors[index]

                elif index == len(colors):
                    color = eraser_color

                elif index == len(colors) + 1:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)

 
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            if y > 80:
                cv2.line(canvas, (prev_x, prev_y), (x, y), color, brush_size)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0

    # merge canvas + camera
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.putText(img, "Colors | E=Eraser | C=Clear | Q=Quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    cv2.imshow("Air Drawing - Full Toolset", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
