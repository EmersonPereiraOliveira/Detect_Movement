import cv2
import numpy as np
import datetime

cap = cv2.VideoCapture(1)
vec1 = []
vec2 = []


#Initialize with the value "None"
first_frame = None
min,max = 0, 0


while(True):
    ret, frame = cap.read()
    if ret == False:
        print("Turn on your camera!")
        break

    text = "Waiting..."
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #To make the image more easy to work
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    #Verify
    if first_frame is None:
        first_frame = gray
        continue

    #Compute the absolute difference between the current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    #cv2.imshow("delta", frame_delta)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("thresh", thresh)

    #Dilate the thresholded image to fill in holes. If bigger Iterations bigger dilate
    thresh = cv2.dilate(thresh, None, iterations=50)
    cv2.imshow("thresh", thresh)

    (_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it. Return the lenght of pointsq
        if cv2.contourArea(c) < 500:
            print(cv2.contourArea(c))
            continue

        # compute the bounding box for the contour, draw it on the frame, and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Objeto: " + str(len(cnts)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        text = "Detected!"

        ####
        vec1.append(int((x + x + w) / 2))
        vec2.append(int((y + y + h) / 2))
        max += 1

    if max > 2:
        for i in range(0, len(vec1), 1):
            cv2.circle(frame, (vec1[i], vec2[i]), 5, (0, 0, 255), -1)


    # draw the text and timestamp on the frame
    cv2.putText(frame, "Status of camera: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Vision of place", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(vec)
cap.release()
cv2.destroyAllWindows()
