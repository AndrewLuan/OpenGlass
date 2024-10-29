import cv2

img_path = 'imageretrieval/photo_24.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img,(100,100))

template_path = 'imageretrieval/photo_25.jpg'
template = cv2.imread(template_path)
template = cv2.resize(template,(100,100))

#灰度图
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
template_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
 
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

#匹配相似度
res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
 
print(res[0][0])