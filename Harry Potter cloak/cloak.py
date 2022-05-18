import cv2
import numpy


def hi(x):
    print ("")


cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("bars")

cv2.createTrackbar("upper_hue","bars",120,180,hi)
cv2.createTrackbar("upper_saturation","bars",255,255,hi)
cv2.createTrackbar("upper_value","bars",255,255,hi)
cv2.createTrackbar("lower_hue","bars",58,180,hi)
cv2.createTrackbar("lower_saturation","bars",55,255,hi)
cv2.createTrackbar("lower_value","bars",54,255,hi)


while(True):
    cv2.waitKey(1000)
    ret, init_frame = cap.read()

    if(ret):
        break

while(True):
    ret, frame = cap.read()    
    inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    upper_hue = cv2.getTrackbarPos("upper_hue","bars")
    upper_saturation = cv2.getTrackbarPos("upper_saturation","bars")
    upper_value = cv2.getTrackbarPos("upper_value","bars")
    lower_hue = cv2.getTrackbarPos("lower_hue","bars")
    lower_saturation = cv2.getTrackbarPos("lower_saturation","bars")
    lower_value = cv2.getTrackbarPos("lower_value","bars")

    kernel = numpy.ones((5,5), numpy.uint16)

    upper_hsv = numpy.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = numpy.array([lower_hue, lower_saturation, lower_value])

    mask = cv2.inRange(inspect, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask,3)
    mask_inverse = 255 - mask
    mask = cv2.dilate(mask, kernel, 5)

    b = frame[:,:,0]
    g = frame[:,:,1]
    r = frame[:,:,2]

    b = cv2.bitwise_and(mask_inverse, b)
    g = cv2.bitwise_and(mask_inverse, g)
    r = cv2.bitwise_and(mask_inverse, r)
    frame_inverse = cv2.merge((b,g,r))

    b = init_frame[:,:,0]
    g = init_frame[:,:,1]
    r = init_frame[:,:,2]

    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    blanket = cv2.merge((b,g,r))


    final = cv2.bitwise_or(frame_inverse,blanket)

    cv2.imshow("Harry's Invisible cloak", final)

    if(cv2.waitKey(3) == ord('q')):
        break;



cv2.destroyAllWindows()
cap.release()        