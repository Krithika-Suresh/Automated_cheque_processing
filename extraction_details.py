import numpy as np
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd='C:\\Users\\Fluffy\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'


def ifsc_extract(image):
    line = image[75:115,150:450]
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    ifsc_num = pytesseract.image_to_string(gray)
    ifsc = ifsc_num.split()
    code = ''
    # print(ifsc)
    for number in ifsc:
        elements = number.replace(',','').replace(':','')
        # elements = number.replace(':','')
        # print(elements)
        if len(elements)==11 and elements.isalnum()==True:
            print("IFSC code is: ",elements)
            code = elements
    account_num(code,image)
    # cv2.imshow("Image", gray)
    # cv2.waitKey(0)
def account_num(code,image):
    line = image[260:300,115:450]
    area = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    # print(code[:5])
    if code[:5]!='SYNB':
        # print("Yes")
        acc_num = pytesseract.image_to_string(area)
        print("Account number is: ", acc_num.split()[0])
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([103,79,60])
        upper = np.array([129,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10:
                cv2.drawContours(mask, [c], -1, (0,0,0), -1)

        mask = 255 - mask
        mask = cv2.GaussianBlur(mask, (3,3), 0)

        data = pytesseract.image_to_string(mask, lang='eng',config='--psm 6')
        print("Account number is: ", data)


image_path = input("Enter the path of the cheque image: ")
image = cv2.imread(image_path)
image = cv2.resize(image, (960, 540))
ifsc_extract(image)

