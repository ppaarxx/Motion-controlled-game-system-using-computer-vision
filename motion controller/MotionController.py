import math
import keyinput
import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX

# 0 For webcam input:
cap = cv2.VideoCapture(1)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read(cv2.WINDOW_NORMAL)
    fp = cv2.flip(image, 1)  # --------> flipped_fp

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    
    # Displaying background colour for fps 
    cv2.rectangle(fp, (0, 0), (120,40), (0,0,0), -1)
     
    # Displaying FPS on the image     -------> Frames per second
    cv2.putText(fp, str(int(fps))+" FPS", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the fp as not writeable to
    fp.flags.writeable = False
    fp = cv2.cvtColor(fp, cv2.COLOR_BGR2RGB)
    results = hands.process(fp)
    fpHeight, fpWidth, _ = fp.shape
   

    # Draw the hand annotations on the fp.
    fp.flags.writeable = True
    fp = cv2.cvtColor(fp, cv2.COLOR_RGB2BGR)
    co=[]
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            fp,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        for point in mp_hands.HandLandmark:
           if str(point) == "HandLandmark.WRIST":
              normalizedLandmark = hand_landmarks.landmark[point]
              pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                        normalizedLandmark.y,
                                                                                    fpWidth, fpHeight)

              try:
                co.append(list(pixelCoordinatesLandmark))
              except:
                  continue

    if len(co) == 2:
        # print(co)
        xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
        radius = 150
        try:
            m=(co[1][1]-co[0][1])/(co[1][0]-co[0][0])
        except:
            continue
        a = 1 + m ** 2
        b = -2 * xm - 2 * co[0][0] * (m ** 2) + 2 * m * co[0][1] - 2 * m * ym
        c = xm ** 2 + (m ** 2) * (co[0][0] ** 2) + co[0][1] ** 2 + ym ** 2 - 2 * co[0][1] * ym - 2 * co[0][1] * co[0][
            0] * m + 2 * m * ym * co[0][0] - 22500

        # centre horizontal line or diameter of the circle
        xa = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        xb = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        ya = m * (xa - co[0][0]) + co[0][1]
        yb = m * (xb - co[0][0]) + co[0][1]
        
        if m!=0:
          ap = 1 + ((-1/m) ** 2)
          bp = -2 * xm - 2 * xm * ((-1/m) ** 2) + 2 * (-1/m) * ym - 2 * (-1/m) * ym
          cp = xm ** 2 + ((-1/m) ** 2) * (xm ** 2) + ym ** 2 + ym ** 2 - 2 * ym * ym - 2 * ym * xm * (-1/m) + 2 * (-1/m) * ym * xm - 22500
          try:
           xap = (-bp + (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
           xbp = (-bp - (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
           yap = (-1 / m) * (xap - xm) + ym
           ybp = (-1 / m) * (xbp - xm) + ym
        
          except:
              continue

        cv2.circle(img=fp, center=(int(xm), int(ym)), radius=radius, color=(15,185,255), thickness=15)

        l = (int(math.sqrt((co[0][0] - co[1][0]) ** 2 * (co[0][1] - co[1][1]) ** 2)) - 150) // 2
        cv2.line(fp, (int(xa), int(ya)), (int(xb), int(yb)), (15,185,255), 20)
        

        if co[0][0] < co[1][0] and co[0][1] - co[1][1] > 65:
            # When the slope is negative, we turn left.
            print("Turning Left")
            keyinput.release_key('s')
            keyinput.release_key('a')
            keyinput.press_key('a')
            cv2.putText(fp, "Turning Left", (130,30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(fp, (int(xap), int(yap)), (int(xm), int(ym)), (255, 0, 0), 20)

        elif co[1][0] > co[0][0] and co[1][1] - co[0][1] > 65:
            print("Turning Right")
            keyinput.release_key('s')
            keyinput.release_key('d')
            keyinput.press_key('d')
            cv2.putText(fp, "Turn Right", (130,30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(fp, (int(xbp), int(ybp)), (int(xm), int(ym)), (255, 0, 0), 20)

        else:
            print("keeping straight")
            keyinput.release_key('s')
            keyinput.release_key('a')
            keyinput.release_key('d')
            keyinput.press_key('w')
            cv2.putText(fp, "keep straight", (130,30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            if ybp>yap:
                cv2.line(fp, (int(xbp), int(ybp)), (int(xm), int(ym)), (15,185,255), 20)
            else:
                cv2.line(fp, (int(xap), int(yap)), (int(xm), int(ym)), (15,185,255), 20)

    if len(co)==1:
       print("Reverse")
       keyinput.release_key('a')
       keyinput.release_key('d')
       keyinput.release_key('w')
       keyinput.press_key('s')
       cv2.putText(fp, "Reverse", (130,30), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Motion controlled game system using computer vision', fp)
    

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()

