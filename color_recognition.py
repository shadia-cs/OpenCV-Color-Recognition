import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colors = [

        ("GREEN",
         np.array([40,40,40]),
         np.array([80,255,255]),
         (0,255,0)),

        ("BLUE",
         np.array([100,150,0]),
         np.array([140,255,255]),
         (255,0,0)),

        ("RED",
         np.array([0,120,70]),
         np.array([10,255,255]),
         (0,0,255))

    ]

    for name, lower, upper, boxColor in colors:

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 1000:

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(frame,
                              (x,y),
                              (x+w,y+h),
                              boxColor,
                              3)

                cv2.putText(frame,
                            name,
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            boxColor,
                            2)

    cv2.imshow("Color Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()