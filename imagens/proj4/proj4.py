import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
import sys
from argparse import RawTextHelpFormatter
from textwrap import wrap

warnings.simplefilter("ignore")

def plotAndSave(img, nome = "NaN", func = "NaF"): #save images
    cv2.imwrite(f"{func}{nome}.png", img)


def encode(img, msg):

    final = img.copy()

    print(final[0,1][1])

    msg = char_generator(msg)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for rgb in range(3):
                pixel_val = img[y, x][rgb]
                try:
                    char = next(msg)
                    if(char=='1'):
                        if(pixel_val%2==0):# eh par entao tem que somar 1
                            final[y,x][rgb] += 1
                    else: #entao o char eh 0, soh muda nos impa
                        if(char=='0'):
                            if(pixel_val%2!=0):
                                final[y,x][rgb] -= 1
                except StopIteration:
                    print("\nfinal\n")
                    print(final)
                    return final

def convert_string(msg):
    msg = wrap(msg, 8)
    print(msg)

    decoded_msg = []

    for substring in msg:
        decoded_msg.append(chr(int(substring, 2)))
    
    decoded_msg = ''.join(decoded_msg)
    print(decoded_msg)

def decode(img):

    msg_bin = []
    count=0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for rgb in range(3):
                pixel_val = img[y, x][rgb]
                if(pixel_val%2==0):
                    msg_bin.append('0')
                    count+=1
                    if(count==9):
                        msg_bin = ''.join(msg_bin)
                        print(msg_bin)
                        convert_string(msg_bin)
                        return
                else:
                    msg_bin.append('1')
                    count=0
    

def char_generator(msg):
    for c in msg:
        yield c

if __name__ == '__main__':
    
    #Getting arguments and reading image
    
    image = sys.argv[1]
    print(image)
    text = sys.argv[2]
    print(text)

    f = open(text, "r")
    if f.mode=='r':
        contents = f.read()
        print(contents)
    
    
    msg = []
    for c in contents:
        msg.append('{0:08b}'.format(ord(c)))

    msg = ''.join(msg)
    print(msg)

    img = cv2.imread("images/baboon.png", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #now it is RGB
    img = img[5: 9, 5:9]
    print(img)
  
    if len(contents) * 8 > img.shape[0]*img.shape[1]*3:
        raise ValueError('There is no room to encode your whole message!')

    final = encode(img, msg)

    decode(final)