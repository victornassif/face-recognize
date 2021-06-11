##Gabriel de Sousa Gomes - RM: 79899
##Lucas Barreto Alencar de Moraes - RM: 80511
##Victor Nassif Marques da Cruz - RM: 80207

import cv2
import numpy as np
import matplotlib.pyplot as plt

#CONST
cascPath = "haarcascade_frontalface_alt.xml"
eyePath = "haarcascade_eye_tree_eyeglasses.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
eyes_cascade = cv2.CascadeClassifier(eyePath)
anterior = 0
Kernal = np.ones((3, 3), np.uint8)#Declare kernal for morphology
croppedScale = 0.11 #

video_capture = cv2.VideoCapture(0)
fig = plt.figure()

x1 = 600
y1 = 800

line1, = plt.plot(x1, y1, 'ko-') 

while True:
    #region Processamento camera
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    ret, frame = video_capture.read()
    frame = cv2.flip(frame,+1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecta faces
    faces = face_cascade.detectMultiScale(
        gray, 1.2, 1
    )

    #Desenha ROI face
    for (face_x, face_y, face_w, face_h) in faces:
        cv2.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)

        faceROI = gray[face_y:face_y+face_h,face_x:face_x+face_w]
        
        eyeLeft = [0,0]
        eyeRight = [0,0]

        #detecta olhos
        eyes = eyes_cascade.detectMultiScale(faceROI, 1.2, 1)
        for (eye_x,eye_y,eye_w,eye_h) in eyes:
            
            #desenha ROI olho
            eye_center = (face_x + eye_x + eye_w//2, face_y + eye_y + eye_h//2)
            radius = int(round((eye_w + eye_h)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            
            ##determina se o olho é esquerdo ou direito
            #TODO: LÓGICA DE VERIFICAR E COMPARAR O X DO OLHO ESQUERDO COM O DIREITO
            if eyeRight[0] < eyeLeft and eyeRight[0] > 0:
                eyeRight[0] = cX
                eyeRight[1] = cY
                
            if eyeLeft[0] < eye_x:
                eyeLeft[0] = cX
                eyeLeft[1] = cY

            #encontra íris
            eyeROI = gray[face_y+eye_y:face_y+eye_y+eye_h, face_x+eye_x:face_x+eye_x+eye_w]
            ret, binary = cv2.threshold(eyeROI, 60, 255, cv2.THRESH_BINARY_INV)
            width, height = binary.shape
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, Kernal)
            dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, Kernal)

            binary = binary[int(croppedScale * height):height, :]##Corta percentual da imagem, para remover sobrancelha
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:
                cnt = contours[0]
                M1 = cv2.moments(cnt)
                
                Cx1 = int(M1['m10'] / M1['m00'])
                Cy1 = int(M1['m01'] / M1['m00'])
                croppedImagePixelLength = int(croppedScale*height)

                cX = int(Cx1+face_x+eye_x)
                cY = int(Cy1+face_y + eye_y + croppedImagePixelLength)

                ##coordenadas íris
                center1 = (cX,cY)
               
                ##desenha íris
                cv2.circle(frame, center1, 2, (0, 255, 0), 2)


                #Insere coordenada no gráfico e desenha para onde está apontando
                line1.set_ydata(eyeLeft)                
                fig.canvas.draw()
            
                #converte canvas para imagem
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
                #Exibe tela com coordenada
                cv2.imshow("plot",img)
                

        if cv2.waitKey(1) & 0xFF == ord('p'):
            print(center1)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #endregion

video_capture.release()
cv2.destroyAllWindows()