import cv2
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from argparse import RawTextHelpFormatter

def get_parser():
    parser = argparse.ArgumentParser(description='Esteanografia', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', default = 'imagens/',
                        help='Folder where the image(s) are')
    parser.add_argument('--msg', help = 'Text file where your message is')
    parser.add_argument('--bit', type=int, choices = range(0,3), default = 0, help = 'Bit layer where you want to hide your message')
    parser.add_argument('--image', default = "baboon", help = 'Choose name of the PNG image you want to run')
    arguments = parser.parse_args()
    return arguments

def plotAndSave(img, name = "NaN", func = "NaF"):#Save images
    cv2.imwrite(f"{func}{name}.png", img)

def encode(img, msg, name="NaN", bit_plane=0):#Encode message char by char
    final = img.copy()
    msg = char_generator(msg)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for rgb in range(3):
                pixel_val = img[y, x][rgb]#Pixel value at layer rgb
                pixel_val_bin = '{0:08b}'.format(pixel_val)#Pixel_val is now binary with 8 bits
                pixel_bit = pixel_val_bin[7-bit_plane]#Pixel we need to check if it needs to be changed
                try:
                    char = next(msg)
                    if(char=='1'):
                        if(pixel_bit=='0'):#If there is a 0 instead of a 1, change it
                            pixel_val_bin = pixel_val_bin[:7-bit_plane] + char + pixel_val_bin[8-bit_plane:]
                            final[y,x][rgb] = int(pixel_val_bin,2)
                    else:#Same pattern as above
                        if(char=='0'):
                            if(pixel_bit=='1'):
                                pixel_val_bin = pixel_val_bin[:7-bit_plane] + char + pixel_val_bin[8-bit_plane:]
                                final[y,x][rgb] = int(pixel_val_bin,2)
                except StopIteration:
                    plotAndSave(cv2.cvtColor(final, cv2.COLOR_RGB2BGR), name, f"encoded_{bit_plane}_")
                    return final

def char_generator(msg):#Generate each character of image when needed
    for c in msg:
        yield c

def get_contents(text):#Read file
    f = open(text, "r")
    if f.mode=='r':
        contents = f.read()
        f.close()
    return contents

def generate_bitplane(im_out, title = ""):  
    plt.axis("off")
    plt.title(title)
    plt.imshow(im_out, cmap="gray", vmin=0, vmax=255)

    plt.waitforbuttonpress(0)
    plt.close("all")

    cv2.imwrite(title + ".png", im_out)

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
    
    contents = get_contents(text)#Getting contents of file to know if there's a \0 at the end to know when to stop
    
    if(ord(contents[-1])!=0):#If not, insert \0
        f = open(text, 'a')
        f.write('\0')
        f.close()
    
    contents = get_contents(text)
    
    msg = []
    for c in contents:
        msg.append('{0:08b}'.format(ord(c)))#Translate content to binary

    msg = ''.join(msg)#Concatenate whole message

    img = cv2.imread("images/" + image_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Now it is RGB
  
    if len(contents) * 8 > img.shape[0]*img.shape[1]*3:
        raise ValueError('There is no room to encode your whole message!')

    image_name = image_name[:-4]
    img_encoded = encode(img, msg, image_name, bit_plane)

    print(f"Sucessfully encoded {image_name}.png!")

    sets = set([0,1,2,7])
    matrixR, matrixG, matrixB = cv2.split(img_encoded)
    for plane in range(0, 8):
        r_out = (matrixR >> plane) & 1
        g_out = (matrixG >> plane) & 1
        b_out = (matrixB >> plane) & 1
        if(plane in sets):
            title = f"r_{plane}_hidein_{bit_plane}_"
            plotAndSave(np.where(r_out, 255, 0), image_name, title)
            title = f"g_{plane}_hidein_{bit_plane}_"
            plotAndSave(np.where(g_out, 255, 0), image_name, title)
            title = f"b_{plane}_hidein_{bit_plane}_"
            plotAndSave(np.where(b_out, 255, 0), image_name, title)

if __name__ == '__main__':
    Main()