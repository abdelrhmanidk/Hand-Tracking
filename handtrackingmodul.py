import cv2
import mediapipe as mp
import time


class handDetection():
     
    def __init__(self, mode=False, maxHands=2, detectioncon=0.5, trackcon=0.5):
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        
        # Initialize the MediaPipe Hands solution.
        self.mp_hands = mp.solutions.hands

        # Create a Hands object with named parameters.
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
                                         max_num_hands=self.maxHands,
                                         min_detection_confidence=self.detectioncon,
                                         min_tracking_confidence=self.trackcon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self,img,draw=True):
                            
            imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results =self.hands.process(imgRGB)
                    # print(results.multi_hand_landmarks)

            if self.results.multi_hand_landmarks:
                     for handLM in self.results.multi_hand_landmarks:
                         if draw:
                                
                       
                             self.mpDraw.draw_landmarks(img, handLM, self.mp_hands.HAND_CONNECTIONS)

            return img
    

    def findPostion(self,img, handNo=0, draw=True):
                  lmList =[]
                  if self.results.multi_hand_landmarks:
                         myhand = self.results.multi_hand_landmarks[handNo]

                         for id,lm in enumerate(myhand.landmark):
                            # print(id,lm)
                            h, w, c = img.shape
                            cx,cy = int(lm.x*w), int(lm.y*h)
                            # print(id,  cx,  cy)
                            lmList.append([id,cx,cy])
                            if draw:
                           
                                cv2.circle(img, (cx,cy),25,(255,0,255),cv2.FILLED)

                  return lmList




def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetection()

    while True:
        success, img = cap.read()
        img = detector.findHand(img)
        lmList = detector.findPostion(img)
        if len(lmList) != 0:
            print(lmList[4])

        # show fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
        main()