from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import requests
import json
import re

from array import array
import os
from PIL import Image, ImageDraw, ImageFont
import sys
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
ENDPOINT = os.getenv('ENDPOINT')
cv_client = ComputerVisionClient(ENDPOINT,CognitiveServicesCredentials(API_KEY))

# image_url = 'https://i.redd.it/pf440nc480921.jpg'
# local_file = 'C:/Users/Fluffy/Downloads/iloveimg-converted/cheque_309062.jpg'
# local_file = 'K:\sem_5\DBMS\project\Automated_cheque_processing\Cheque_Image_Dataset\cheques\cheque_309075.jpg'
local_file = 'K:/sem_5/DBMS/project/Automated_cheque_processing/Cheque_Image_Dataset/cheques/cheque_309062.jpg'
# respnse = cv_client.read(url=image_url,language='en',raw=True)
response = cv_client.read_in_stream(open(local_file,'rb'), language='en', raw=True)
# print(response)
operationLocation = response.headers['Operation-Location']
operation_id = operationLocation.split('/')[-1]
time.sleep(5)
result = cv_client.get_read_result(operation_id)
# print(result)
print(result.status)
lst = []
lst_1 = []
if result.status == OperationStatusCodes.succeeded:
    read_results = result.analyze_result.read_results
    for analyzed_result in read_results:
        # l = len(analyzed_result.lines)
        # print(l)
        # lst_name = ['DadabaD. Pradeep Kumar', 'Abhilash Reddy', 'G. Jayakrishna Reddy', 'J. Ravi Kumar Singh']
        for line in analyzed_result.lines:
            lst_1.append(line.text)
        #     if line.text == 'SyndicateBank':
        #         lst.append(line.text)
        #     elif 'IFSC' in line.text:
        #         lst.append(line.text[7:])
        #     elif (line.text).isdigit() == True and len(line.text)==8:
        #         lst.append(line.text)
        #     elif line.text in lst_name:
        #         lst.append(line.text)
        #     elif '/-' in line.text:
        #         lst.append(line.text)
        #     elif len(line.text)>=14:
        #         x= (line.text).replace(' ',"")
        #         x= x.replace('_',"")
        #         # print(x)
        #         if x.isdigit()==True:
        #             # print(x)
        #             lst.append(x)
########################################################################################################################################
            # lst_name = ['B. Shiva Kumar.', 'Ravuri Manish Olha', 'Kumar Ravi Palli', 'Rajesh Amit Som','V. Tesaswwnam Krishna', 'B. Shranan Kumar','Kishore Ras Shukla']
            # if line.text == 'ICICI Bank' or line.text=="İICICI Bank":
            #     if line.text=='ICICI Bank':
            #         lst.append(line.text)
            #     else:
            #         lst.append(line.text[1:])
            # elif 'IFSC' in line.text:
            #         # print(line.text[24:])
            #         lst.append(line.text[24:])
                
            # elif (line.text).isdigit() == True and len(line.text)==8:
            #     lst.append(line.text)
            # elif line.text in lst_name:
            #     lst.append(line.text)
            # elif '/-' in line.text:
            #     lst.append(line.text)
            # elif len(line.text)==12 and (line.text).isdigit()==True:
            #         lst.append(line.text)
# ##########################################################################################################################################
            if line.text == 'AXIS BANK LTD':
                lst_name = ['K. Rajesh Gowd', 'Vijay Kumar Singh', 'Amita Kadam', 'Banother Latika Makhija', 'Sunil Kunas','Chahat Thalwar', 'Rohit Kanna','G. PRAFULLA SARANGI','R. Ravi Sudhakan Reddy','Raju Maiti','R. Kumar Deuck','P.D. Ravi. Vamsi']
                lst.append(line.text)
            elif 'IFS CODE' in line.text:
                    # print(line.text[24:])
                    lst.append(line.text[11:])
                
            elif (line.text).isdigit() == True and len(line.text)==8:
                lst.append(line.text)
            elif line.text in lst_name:
                lst.append(line.text)
            elif '₹' in line.text:
                lst.append(line.text)
            elif len(line.text)==15 and (line.text).isdigit()==True:
                    lst.append(line.text)

            # print('Line: ', line.text)
            
    lst.append(lst_1[-1])
    print(lst)