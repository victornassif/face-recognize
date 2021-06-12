import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10,7.5)) 
# plt.axis([0, 800, 0, 600])

# plt.xlim(0, 800)
# plt.ylim(0, 600)
cap = cv2.VideoCapture(0)

# x1 =  np.linspace(1000.0,1000.0)
# y1 =  np.linspace(0.0, 1000.0)

# line1, = plt.plot(x1, y1, 'ko-')        # so that we can update data later

for i in range(1000):
    plt.xlim(0, 800)
    plt.ylim(0, 600)
    # update data
    plt.scatter(i,i)
    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


    # display image with opencv or any operation you like
    cv2.imshow("plot",img)

    # display camera feed
    ret,frame = cap.read()
    cv2.imshow("cam",frame)

    plt.clf()

    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break

# import numpy as np
# import matplotlib.pyplot as plt
# from random import randrange



# y = 10
# isUp = True
# for i in range(0, 800, 10):
#     #y = randrange(600)
#     if y >= 600:
#         isUp = False
#     if y < 10:
#         isUp = True

#     if isUp:
#         y += 10
#     else:
#         y -= 10
#     plt.xlim(0, 800)
#     plt.ylim(0, 600)
#     plt.scatter(i, y)
#     plt.scatter(i + 100, y)
#     plt.pause(0.05)
#     plt.clf()

# plt.show()