import cv2
import numpy as np
import sys
import argparse
from argparse import RawTextHelpFormatter
from textwrap import wrap

def get_parser():
    parser = argparse.ArgumentParser(description='Esteanografia', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', default = 'imagens/',
                        help='Folder where the image(s) are')
    parser.add_argument('--msg', default = 'texto_saida.txt', help = 'Text file where your message will be decoded')
    parser.add_argument('--bit', type=int, choices = range(0,3), default = 0, help = 'Bit layer where your'
                        + ' hidden message is')
    parser.add_argument('--image', default = "baboon", help = 'Choose name of the PNG image you want to run')
    arguments = parser.parse_args()
    return arguments


def convert_string(msg, text, bit_plane):
    msg = wrap(msg, 8)#Split the whole string into 8 bit chunks
    decoded_msg = []

    for substring in msg:#For each substring, convert then to decimal and get your ASCII entry 
        decoded_msg.append(chr(int(substring, 2)))
    
    decoded_msg = ''.join(decoded_msg)#Get the message and print it
    print("The decoded message is: ")
    print(decoded_msg)#Final print
    
    f = open(f"{bit_plane}_" + text, "w+")
    f.write(decoded_msg)
    f.close()


def decode(img, text, bit_plane=0):
    msg_bin = []
    count=0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for rgb in range(3):
                pixel_val = img[y, x][rgb]
                pixel_val_bin = '{0:08b}'.format(pixel_val)#Pixel_val is now binary with 8 bits
                if(pixel_val_bin[7-bit_plane]=='0'):#Just get the least significant bit of each layer of each pixel
                    msg_bin.append('0')
                    count+=1
                    if(count>=9 and len(msg_bin)%8==0):
                        msg_bin = msg_bin[:-8]
                        msg_bin = ''.join(msg_bin)#Concatenate everything when finding 8 straight 0's representing '\0'
                        convert_string(msg_bin, text, bit_plane)
                        return
                else:
                    msg_bin.append('1')
                    count=0

def Main():
    #Reading image and text file
    arguments = get_parser()
    image_name = arguments.image
    text = arguments.msg
    bit_plane = int(arguments.bit)

    if '.png' not in image_name:
        image_name += '.png'
    if '.txt' not in text:
        text += '.txt'
    
    img = cv2.imread("encoded_" + image_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Now it is RGB
  
    decode(img, text, bit_plane)

if __name__ == '__main__':
    Main()