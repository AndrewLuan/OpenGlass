import rpa as r
from pyzbar.pyzbar import decode
from PIL import Image
from io import BytesIO
import win32clipboard
import cv2
def send_msg_to_clip(type_data, msg):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(type_data, msg)
    win32clipboard.CloseClipboard()
def paste_img(file_img):
    image = Image.open(file_img)
    output = BytesIO()
    image.save(output, 'BMP')
    data = output.getvalue()[14:]
    output.close()
    send_msg_to_clip(win32clipboard.CF_DIB, data)

def qrauto(file_image):
    image = cv2.imread(file_image)
    text_content = decode(image)
    if(len(text_content)!=0):
        url = text_content[0].data.decode('utf-8')
        if(url[:20] == 'https://u.wechat.com'):
            paste_img(file_image)
            r.init(visual_automation = True,chrome_browser=False)
            # r.click('images/wechat.png')
            r.keyboard('[ctrl][alt][w]')
            r.keyboard('[ctrl][v]')
            r.keyboard('[enter]')
            r.hover('images/send.png')
            r.click(r.mouse_x(),r.mouse_y()-300)
            r.click('images/point.png')
            r.click('images/decode.png')
            r.click('images/add.png')
            r.click('images/confirm.png')
            r.close()
        else:
            meassage = url.split('\n')
            if(meassage[1]=='music'):
                r.init(visual_automation = True)
                r.url(meassage[0])
                r.click('//*[@id="appRoot"]/div/div[3]/div[4]/div/div/div[3]/div[2]/div/div[3]/div[2]/div/div/div[2]/span[2]/button/i')
                # r.close()
            elif(meassage[1]=='video'):
                r.init(visual_automation = True)
                r.url(meassage[0])
                r.click('//*[@id="__bolt-photo-main"]/div[1]/div[2]/div[1]/video')
                # r.close()
if __name__ == "__main__":
    file_image = 'images/video1.png'
    qrauto(file_image)
    # file_image = 'images/music1.png'
    # qrauto(file_image)
    # file_image = 'images/testqr1.jpg'
    # qrauto(file_image)