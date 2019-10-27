import cv2
import numpy as np
import sys
from textwrap import wrap

def convert_string(msg):
    msg = wrap(msg, 8)#Split the whole string into 8 bit chunks
    decoded_msg = []

    for substring in msg:#For each substring, convert then to decimal and get your ASCII entry 
        decoded_msg.append(chr(int(substring, 2)))
    
    decoded_msg = ''.join(decoded_msg)#Get the message and print it
    print("The decoded message is: ")
    print(decoded_msg)#Final print

def decode(img):
    msg_bin = []
    count=0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for rgb in range(3):
                pixel_val = img[y, x][rgb]
                if(pixel_val%2==0):#Just get the least significant bit of each layer of each pixel
                    msg_bin.append('0')
                    count+=1
                    if(count==9):
                        msg_bin = ''.join(msg_bin)#Concatenate everything when finding 8 straight 0's representing '\0'
                        convert_string(msg_bin)
                        return
                else:
                    msg_bin.append('1')
                    count=0

def Main():
    #Reading image and text file
    image_name = sys.argv[1]
    text = sys.argv[2]

    if '.png' not in image_name:
        image_name += '.png'
    
    img = cv2.imread("encoded_" + image_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Now it is RGB
  
    decode(img)

if __name__ == '__main__':
    Main()