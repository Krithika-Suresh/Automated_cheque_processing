from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):

    charIter = charCnts.__iter__()
    rois = []
    locs = []
    while True:
        try:
			
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None
            if cW >= minW and cH >= minH:
				# extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
				
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,-np.inf)
				
                for p in parts:
					
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)
				# extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
        except StopIteration:
            break
    return (rois, locs)

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
	help="K:\sem_5\DBMS\project\cheque_processing\IDRBT Cheque Image Dataset\cheques\cheque_100834.tif")
ap.add_argument("-r", "--reference", required=True,
	help="K:\sem_5\DBMS\project\cheque_processing\micr_chars.png")
args = vars(ap.parse_args())

# initialize the list of reference character names, in the same
# order as they appear in the reference image where the digits
# their names and:
# T = Transit (delimit bank branch routing transit #)
# U = On-us (delimit customer account number)
# A = Amount (delimit transaction amount)
# D = Dash (delimit parts of numbers, such as routing or account)
charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0","T", "U", "A", "D"]

ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = imutils.resize(ref, width=400)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]


refROIs = extract_digits_and_symbols(ref, refCnts,
	minW=10, minH=20)[0]
chars = {}

for (name, roi) in zip(charNames, refROIs):
	
    roi = cv2.resize(roi, (36, 36)) 
    chars[name] = roi
    # cv2.imshow("Image",roi)
    # cv2.waitKey(0)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
output = []
# cv2.imshow("Image", rectKernel)
# cv2.waitKey(0)

image = cv2.imread(args["image"])
image = cv2.resize(image, (960, 540))
line = image[450:550,230:700]
(h, w,) = image.shape[:2]
# # print(image.shape)
delta = int(h - (h * 0.15))
# # print(h,w,delta)
# bottom = image[delta:h, 0:w-500]
gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# cv2.imshow("Greyscale", blackhat)
# cv2.waitKey(0)

gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
# cv2.imshow("Greyscale", gradX)
# cv2.waitKey(0)

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = clear_border(thresh)
# cv2.imshow("Greyscale", thresh)
# cv2.waitKey(0)

# # ####################################################################################3
groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
groupCnts = imutils.grab_contours(groupCnts)
groupLocs = []
# loop over the group contours
for (i, c) in enumerate(groupCnts):
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)

	if w > 50 and h > 15:
		groupLocs.append((x, y, w, h))

groupLocs = sorted(groupLocs, key=lambda x:x[0])
# print(groupLocs)


for (gX, gY, gW, gH) in groupLocs:
    groupOutput = []

    group = gray[gY -5:gY+gH+5, gX-5:gX+gW+15]
    # group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow("Group", group)
    # cv2.waitKey(0) 

    charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    charCnts = imutils.grab_contours(charCnts)
    charCnts = contours.sort_contours(charCnts,method="left-to-right")[0]
    (rois, locs) = extract_digits_and_symbols(group, charCnts)
    for roi in rois:
        scores = []
        roi = cv2.resize(roi, (36, 36))
        for charName in charNames:
            result = cv2.matchTemplate(roi, chars[charName],cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(charNames[np.argmax(scores)])
    cv2.rectangle(image, (gX - 5, gY + delta - 10),
		(gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
    cv2.putText(image, "".join(groupOutput),
		(gX - 5, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX,
		0.95, (0, 0, 255), 3)
	# add the group output to the overall check OCR output
    output.append("".join(groupOutput))
print("Cheque MICR number: {}".format(" ".join(output)))
# cv2.imshow("Check OCR", image)
# cv2.waitKey(0)