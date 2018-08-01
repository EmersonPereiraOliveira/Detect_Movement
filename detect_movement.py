import cv2
import numpy as np
import datetime


#Função que recebe as coordenadas do objeto e pinta-o
def desenhar_coordenada(x, y):
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

#Função que desenha retangulos
def desenhar_retangulo(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

#Função que recebe uma imagem e retorna apenas a variável com os contornos
# e os momentos, ignorando o restante da informações.
def detectar_contornos(img):
    (_, cnts, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


cap = cv2.VideoCapture(0)
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
    cv2.imshow("Gaus", gray)

    #Verify
    if first_frame is None:
        first_frame = gray
        continue

    #Compute the absolute difference between the current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    #Dilate the thresholded image to fill in holes. If bigger Iterations bigger dilate
    thresh = cv2.dilate(thresh, None, iterations=25)
    cv2.imshow("thresh", thresh)

    cnts = detectar_contornos(thresh.copy())


    contador = 1
    # Loop que percorre cada "contorno" detectado.
    for c in cnts:

        # Se o tamanho do objeto detectado for menor, ignore!
        if cv2.contourArea(c) < 1000:
            print("Menor")
            continue

        #Se o tamanho do objeto detectado for muito grande, ignore!
        #Tentative de correção do bug de inicio com a webcam do not
        if cv2.contourArea(c) > 20000:
            print("Maior")
            continue


        # compute the bounding box for the contour, draw it on the frame, and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        desenhar_retangulo(frame, x,y,w,h)
        cv2.putText(frame, "Objeto: " + str(contador), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        text = "Detected!"
        contador+=1

        ####
        vec1.append(int((x + x + w) / 2))
        vec2.append(int((y + y + h) / 2))
        max += 1

    contador = 0
    if max > 1:
        for i in range(0, len(vec1), 1):
            desenhar_coordenada(vec1[i], vec2[i])

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Status of camera: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Visão do ambiente", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
