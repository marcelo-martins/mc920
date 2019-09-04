import numpy as np;
import cv2;
import math
import matplotlib.pyplot as plt


def goThroughGray(matrix, title, bordersize):            #inverter o sentido da mascara criar borda, ver se funciona com retangulo

    (row,column) = matrix.shape
    print(row, column)
    print(matrix[511,511])
    matrixAux = matrix*0

    row-= bordersize
    column-=bordersize

    #origem fica em cima
    for i in range(bordersize, row):#i row, j column
        if(i%2!=0):
            for j in reversed(range(bordersize, column)):
                if(matrix[i,j] < 128):
                    matrixAux[i,j] = 0
                else:
                    matrixAux[i,j] = 1
                err = matrix[i,j] - matrixAux[i,j]*255
                matrix[i, j - 1] += (7/16)*err
                matrix[i+1,j+1] += (3/16)*err
                matrix[i+1, j] += (5/16)*err
                matrix[i+1,j-1] += (1/16)*err
            
        else:
            for j in range(bordersize, column):
                if(matrix[i,j] < 128):
                    matrixAux[i,j] = 0
                else:
                    matrixAux[i,j] = 1
                err = matrix[i,j] - matrixAux[i,j]*255
                matrix[i, j + 1] += (7/16)*err
                matrix[i+1,j-1] += (3/16)*err
                matrix[i+1, j] += (5/16)*err
                matrix[i+1,j+1] += (1/16)*err


    plt.axis("off")
    plt.imshow(matrixAux, cmap="gray", vmin=0, vmax=1)
    plt.title(title)

    plt.waitforbuttonpress(0)
    plt.close("all")

    cv2.imwrite(title+".png", matrixAux)
    return matrixAux



def testColor():
   
    matrixColor = cv2.cvtColor(cv2.imread("baboon.png"), cv2.COLOR_BGR2RGB)
   
    
    matrixB, matrixG, matrixR = cv2.split(matrixColor)

    bordersize=5
    matrixR = cv2.copyMakeBorder(matrixR, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixG = cv2.copyMakeBorder(matrixG, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixB = cv2.copyMakeBorder(matrixB, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])

    plt.axis("off")
    plt.imshow(matrixColor)
    plt.waitforbuttonpress(0)
    plt.close("all")

   
    goThroughGray(matrixG, "color", bordersize)
    final = cv2.merge((matrixB, matrixG, matrixR))
    plt.axis("off")
    plt.imshow(final)
    plt.waitforbuttonpress(0)
    plt.close("all")



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
    testColor()
    
