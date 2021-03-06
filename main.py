# Gabriel de Sousa Gomes - RM: 79899
# Lucas Barreto Alencar de Moraes - RM: 80511
# Victor Nassif Marques da Cruz - RM: 80207

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import eye

# CONST
cascPath = "haarcascade_frontalface_alt.xml"
eyePath =  "haarcascade_eye_tree_eyeglasses.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
eyes_cascade = cv2.CascadeClassifier(eyePath)
anterior = 0
Kernal = np.ones((3, 3), np.uint8)# Declare kernal for morphology
croppedScale = 0.11

video_capture = cv2.VideoCapture(0)
fig = plt.figure(figsize=(10,7.5)) 

while True:
    # region Processamento camera
    if not video_capture.isOpened():
        print('Unable to load camera.')
        pass

    ret, frame = video_capture.read()
    frame = cv2.flip(frame, +1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta faces
    faces = face_cascade.detectMultiScale(
        gray, 1.2, 1
    )

    # Instanciada antes da face, pois a face é instanciada diversas vezes, e fará o ponto central sumir
    lastSeenLeftEye = [0, 0]
    lastSeenRightEye = [0, 0]
    mediaBetweenEyes = [0, 0]

    # Desenha ROI face
    for (face_x, face_y, face_w, face_h) in faces:
        cv2.rectangle(frame, (face_x, face_y),
                      (face_x+face_w, face_y+face_h), (0, 255, 0), 2)

        faceROI = gray[face_y:face_y+face_h, face_x:face_x+face_w]

        leftEye = [0, 0]
        rightEye = [0, 0]
        isLeftEye = False
        isRightEye = False

        # detecta olhos
        eyes = eyes_cascade.detectMultiScale(faceROI, 1.2, 1)
        for (eye_x, eye_y, eye_w, eye_h) in eyes:

            # desenha ROI olho
            eye_center = (face_x + eye_x + eye_w//2, face_y + eye_y + eye_h//2)
            radius = int(round((eye_w + eye_h)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

            # determina se o olho é esquerdo ou direito de acordo com a posição do X na face
            if eye_x >= 80:
                isLeftEye = True
                isRightEye = False
            elif eye_x <= 80 and eye_x > 0:
                isRightEye = True
                isLeftEye = False

            # encontra íris
            eyeROI = gray[face_y+eye_y:face_y+eye_y +
                          eye_h, face_x+eye_x:face_x+eye_x+eye_w]
            ret, binary = cv2.threshold(eyeROI, 60, 255, cv2.THRESH_BINARY_INV)
            width, height = binary.shape
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, Kernal)
            dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, Kernal)

            # Corta percentual da imagem, para remover sobrancelha
            binary = binary[int(croppedScale * height):height, :]
            contours, hierarchy = cv2.findContours(
                dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:
                cnt = contours[0]
                M1 = cv2.moments(cnt)

                Cx1 = int(M1['m10'] / M1['m00'])
                Cy1 = int(M1['m01'] / M1['m00'])
                croppedImagePixelLength = int(croppedScale*height)

                cX = int(Cx1 + face_x + eye_x)
                cY = int(Cy1 + face_y + eye_y + croppedImagePixelLength)

                # coordenadas íris
                center1 = (cX, cY)

                # desenha íris
                cv2.circle(frame, center1, 2, (0, 255, 0), 2)

                if isLeftEye and cX > 0:
                    leftEye[0] = cX
                    leftEye[1] = cY
                    lastSeenLeftEye = leftEye
                elif isRightEye and cX > 0:
                    rightEye[0] = cX
                    rightEye[1] = cY
                    lastSeenRightEye = rightEye

                if (lastSeenLeftEye[0] > 0 and lastSeenRightEye[0] > 0):
                    mediaBetweenEyes = [(lastSeenRightEye[0] + lastSeenLeftEye[0])/2, (lastSeenRightEye[1]+lastSeenLeftEye[1])/2]
                    # Insere coordenada no gráfico e desenha para onde está apontando
                    plt.xlim(0, 800)
                    plt.ylim(0, 600)
                    # update data
                    plt.scatter(mediaBetweenEyes[0],mediaBetweenEyes[1])
                    # redraw the canvas
                    fig.canvas.draw()

                    # convert canvas to image
                    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
                    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    # Convert to opencv's default bgr
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


                    # Mostra imagem com OpenCV
                    cv2.imshow("plot",img)
                    plt.clf()

                    print('E' + str(lastSeenRightEye) +' - ' + 'R' + str(lastSeenLeftEye))

        if cv2.waitKey(1) & 0xFF == ord('p'):
            print(center1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # endregion


video_capture.release()
cv2.destroyAllWindows()