import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='Global and local filters testing')
    parser.add_argument('--folder', default = '/home/marcelomartins/Documentos/8sem/imagens/proj2/imagens',
                        help='Folder where the image(s) are')#required = False é um argumento
    parser.add_argument('--option', type=int, choices = range(0,7), 
                        help = 'Choose filter you want to run')

    arguments = parser.parse_args()
    return arguments


def globalMethod(img):# 0 is height, 1 is weight but 0 is y and weight is x

    final = img.copy()    
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            #final = np.where(img < 128, 0, 255)
            if(img[y, x] > 128):
                final[y,x] = 0
            else:
                final[y,x] = 255
    cv2.imshow("nome", final)
    cv2.waitKey(0)


    #final = np.where(final < 128, 0, 255)
    
def niblack(img): #higher than T is black, otherwise is white

    final = img.copy()    
    for y in range(1, img.shape[0]-1): #ignoring borders
        for x in range(1, img.shape[1]-1):
            temp = final[y-1:y+2, x-1:x+2]
            maxValue = np.amax(temp)
            minValue = np.amin(temp)

            result = (maxValue + minValue)/2
            if(img[y, x] > result):
                final[y,x] = 0
            else:
                final[y,x] = 255
    

    cv2.imshow("nome", final)
    cv2.waitKey(0)



if __name__ == '__main__':
    arguments = get_parser()
    print(arguments.option)
    img = cv2.imread('imagens/baboon.pgm', cv2.IMREAD_UNCHANGED)
    print(img.dtype)

    opt = int(arguments.option)
    
    if(opt==1):
        globalMethod(img)
    if(opt==2):
        niblack(img)