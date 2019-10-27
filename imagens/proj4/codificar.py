import cv2
import numpy as np
import sys

def plotAndSave(img, name = "NaN", func = "NaF"):#Save images
    cv2.imwrite(f"{func}{name}.png", img)

def encode(img, msg, name="NaN"):#Encode message char by char
    final = img.copy()
    msg = char_generator(msg)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for rgb in range(3):
                pixel_val = img[y, x][rgb]
                try:
                    char = next(msg)
                    if(char=='1'):
                        if(pixel_val%2==0):#If even and char==1, we need to sum 1 because the last bit has to be 1
                            final[y,x][rgb] += 1
                    else:#If odd and char==0, as above, we need to change last bit to 0 by subtracting 1
                        if(char=='0'):
                            if(pixel_val%2!=0):
                                final[y,x][rgb] -= 1
                except StopIteration:
                    plotAndSave(cv2.cvtColor(final, cv2.COLOR_RGB2BGR), name, "encoded_")
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

def Main():
    #Reading image and text file
    image_name = sys.argv[1]
    text = sys.argv[2]

    if '.png' not in image_name:
        image_name += '.png'
    
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
    encode(img, msg, image_name)

    print(f"Sucessfully encoded {image_name}.png!")

if __name__ == '__main__':
    Main()