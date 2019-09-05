import cv2;
import numpy as np;
import math
import matplotlib.pyplot as plt
import os
import glob

#Getting all files
def getGray():
    image_dir = "/home/marcelomartins/Documentos/8sem/imagens/proj1/images"   
    data_path = os.path.join(image_dir, '*g')
    files = glob.glob("/home/marcelomartins/Documentos/8sem/imagens/proj1/images/*.png")
    data=[]
    names = []
    for fl in files:
        img = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        data.append(img)
        names.append(os.path.splitext(os.path.basename(fl))[0])
    return data, names

def getColor():
    image_dir = "/home/marcelomartins/Documentos/8sem/imagens/proj1/images"
    data_path = os.path.join(image_dir, '*g')
    files = glob.glob("/home/marcelomartins/Documentos/8sem/imagens/proj1/images/*.png")
    data=[]
    names = []
    for fl in files:
        img = cv2.cvtColor(cv2.imread(fl), cv2.COLOR_BGR2GRAY)
        data.append(img)
        names.append(os.path.splitext(os.path.basename(fl))[0])
    return data, names

def plotAndSave(title = ""):  
    plt.axis("off")
    plt.title(title)
    plt.imshow(im_out, cmap="gray", vmin=0, vmax=255)

    plt.waitforbuttonpress(0)
    plt.close("all")

    cv2.imwrite(title+".png", im_out)



def goThroughGray(matrix, title, bordersize, color):            #inverter o sentido da mascara criar borda, ver se funciona com retangulo

    (row,column) = matrix.shape
    print(row, column)
    matrixAux = np.copy(matrix)

    row-= bordersize
    column-=bordersize

    #origem fica em cima
    for i in range(bordersize, row):#i row, j column
        if(i%2!=0):
            for j in reversed(range(bordersize, column)):
                if(matrix[i,j] < 128):
                    matrixAux[i,j] = 0
                else:
                    matrixAux[i,j] = 255
                err = int(matrix[i,j]) - int(matrixAux[i,j])
                err = float(err)
                matrix[i, j - 1] += (7/16)*err
                matrix[i+1,j+1] += (3/16)*err
                matrix[i+1, j] += (5/16)*err
                matrix[i+1,j-1] += (1/16)*err
            
        else:
            for j in range(bordersize, column):
                if(matrix[i,j] < 128):
                    matrixAux[i,j] = 0
                else:
                    matrixAux[i,j] = 255
                err = int(matrix[i,j]) - int(matrixAux[i,j])
                err = float(err)
                matrix[i, j + 1] += (7/16)*err
                matrix[i+1,j-1] += (3/16)*err
                matrix[i+1, j] += (5/16)*err
                matrix[i+1,j+1] += (1/16)*err


    # plt.axis("off")
    # plt.imshow(matrixAux, cmap="gray", vmin=0, vmax=255)
    # plt.title(title)

    # plt.waitforbuttonpress(0)
    # plt.close("all")

    #plt.savefig(title + '.png', bbox_inches='tight')
    return matrixAux



def testColor(title, matrixColor):
   
    #matrixColor = cv2.cvtColor(cv2.imread("train.png"), cv2.COLOR_BGR2RGB)
   
    matrixColor = cv2.cvtColor(matrixColor, cv2.COLOR_BGR2RGB)
    matrixB, matrixG, matrixR = cv2.split(matrixColor)

    bordersize=5
    matrixR = cv2.copyMakeBorder(matrixR, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixG = cv2.copyMakeBorder(matrixG, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixB = cv2.copyMakeBorder(matrixB, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])

    #print da inicial
    # plt.axis("off")
    # plt.imshow(matrixColor)
    # plt.waitforbuttonpress(0)
    # plt.close("all")

    #chama a funcao pras 3
    matrixR = goThroughGray(matrixR, "r", bordersize, "Reds")
    matrixG = goThroughGray(matrixG, "g", bordersize, "Greens")
    matrixB = goThroughGray(matrixB, "b", bordersize, "Blues")


   
    
    final = cv2.merge((matrixB, matrixG, matrixR))
    # plt.title("color")
    # plt.axis("off")
    # plt.imshow(final)
    # plt.waitforbuttonpress(0)
    # plt.close("all")
    cv2.imwrite(title + 'ColorFloydSteinberg.png', cv2.cvtColor(final, cv2.COLOR_BGR2RGB))



def testGray():
    #goThroughGray()
    matrixGray = cv2.cvtColor(cv2.imread("baboon.png"), cv2.COLOR_BGR2GRAY)
    #matrixColor = cv2.imread('baboon.png', cv2.IMREAD_COLOR)
    
    
    # b, g, r = cv2.split(matrixColor)
    # matrixR = goThroughGray(r, "r")
    # matrixG = goThroughGray(g, "g")
    # matrixB = goThroughGray(b, "b")

    bordersize=5
    matrixGray = cv2.copyMakeBorder(matrixGray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])

    plt.axis("off")
    plt.imshow(matrixGray, cmap="gray", vmin=0, vmax=255)
    plt.waitforbuttonpress(0)
    plt.close("all")

   
    goThroughGray(matrixGray, "zinza", bordersize)


if __name__ == '__main__':

    #testGray()
    data, names = getColor()
    i = 0
    for image in data:
        testColor(names[i], image)
        i+=1
    
