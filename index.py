import cv2
import numpy as np

#kullandığımız kameranın giriş bilgileri ve ip adresi
cap = cv2.VideoCapture(0)
assert cap.isOpened()

#Hareketi algılama modülü
object_detector = cv2.createBackgroundSubtractorMOG2()


#sharpness için girilmesi gereken değerler
height_value = 0
width_value = 0
sigmaX_value = 250
sigmaY_value = 250

#brightness için girilmesi gereken değerler
alpha = 1.33
beta = -50

#contrast için girilmesi gereken değerler
contrast = 1
brightness = 0

#foroğrafın ilk değeri
img_counter = 0

#tuş atamaları
while(True):
    ret, frame = cap.read()
    if not ret:
        break

    k = cv2.waitKey(1)
#sharpness ayarları
    if k%256 == ord('q'):
     height_value+= 5
     print(str(height_value) + "height artırıldı")    
         
    elif k%256 == ord('a'):
     height_value -= 5
     print(str(height_value) + "height azaltıldı")

    if k%256 == ord('w'):
     width_value+= 5
     print(str(width_value) + "width artırıldı")    
         
    elif k%256 == ord('s'):
     width_value -= 5
     print(str(width_value) + "width azaltıldı")
     
    #sigma ayarları  
     '''
    elif k%256 == ord('e'):
     sigmaX_value += 10
     print(str(sigmaX_value) + "sigmaX artırıldı")

    elif k%256 == ord('d'):
     sigmaX_value -= 10
     print(str(sigmaX_value) + "sigmaX azaltıldı")

    elif k%256 == ord('r'):
     sigmaY_value += 10
     print(str(sigmaY_value) + "sigmaY artırıldı")

    elif k%256 == ord('f'):
     sigmaY_value -= 10
     print(str(sigmaY_value) + "sigmaY azaltıldı")
     '''
#parlaklık ayarı
    elif k%256 == ord('4'):
     print(str(alpha) + "parlaklık artırıldı")
     alpha+=0.15   
         
    elif k%256 == ord('1'):
     alpha -= 0.15
     print(str(alpha) + "parlaklık azaltıldı")

#contrast ayarı
    if k%256 == ord('5'):
     print(str(contrast) + "kontrast artırıldı")
     contrast+=0.25
         
         
    elif k%256 == ord('2'):
     contrast -= 0.25
     print(str(contrast) + "kontrast azaltıldı")
      
#denklemler
    k_width = width_value *2 + 1
    k_height = height_value *2 + 1

#brightness denklemi
    frame = cv2.addWeighted(frame,alpha,np.zeros(frame.shape,frame.dtype),0,beta)

#contrast denklemi
    frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255 )

#sharpness denklemleri
    frame2 = cv2.GaussianBlur(frame,(k_width,k_height),sigmaX_value,sigmaY_value)

    modified = cv2.addWeighted(frame, 1.5, frame2, -0.5, 0)

#mask adında bir filtre uygulayarak hareketi tespiti
    mask = object_detector.apply(modified)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#hareketli cisimleri kutucuk içine alması için yazılan fonksiyon

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(modified, (x, y), (x + w, y + h), (255, 0, 0), 2)



#eger görüntüyü siyah beyaz görmek istersek
    #modified= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('IP Camera', modified)
    #cv2.imshow('IP Camera', mask)
    
    
#ESC tusu (çıkıs icin)    
    if k%256 == 27:
        break

# SPACE tusu (anlık foto almak için)
    elif k%256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} yazdirildi!".format(img_name))
        img_counter += 1

cap.release()
cv2.destroyAllWindows()