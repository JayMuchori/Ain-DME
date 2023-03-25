import numpy as np
import cv2
from PIL import Image
import base64
import io
def main(image):
  #resize image
  ## scale image
  decoded_data=base64.b64decode(image)
  np_data=np.fromstring(decoded_data,np.uint8)
  img=cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
  img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  #resize image
  ## scale image
  x = img_rgb[img_rgb.shape[0]//2 , : , : ].sum ( 1 )
  r=(x>x.mean()/5 ).sum()/2
  s=300*1.0/ r

  ##resize
  image = cv2.resize(img_rgb, (0,0),fx=s,fy=s)

#subtract local mean color
  a=cv2.addWeighted(image, 4, cv2.GaussianBlur(image,( 0, 0 ), 10), -4,128)
#create a mask
  b=np.zeros(a.shape )
  cv2.circle(b, (a.shape [1]//2 , a.shape [0]//2), int(300*0.9 ), (1,1,1), -1, 8, 0)

#put the mean subtracted image on the mask
  a=a*b+128*(1-b)
  a_black=a*b+0*(1-b)

  pil_im = Image.fromarray(np.uint8(a_black))
  buff=io.BytesIO()
  pil_im.save(buff,format="PNG")
  img_str=base64.b64encode(buff.getvalue())
  return ""+str(img_str,'utf-8')




