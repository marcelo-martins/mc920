import cv2;
import numpy as np;
import math
import matplotlib.pyplot as plt
import os
import glob

floyd = np.array([[0, 0, 7/16],
           [3/16, 5/16, 1/16]]).reshape(2,3)

stevenson = np.array([[0 , 0 , 0 , 0 , 0 , 32/200 , 0],
            [12/200, 0 , 26/200 , 0 , 30/200 , 0 , 16/200],
            [0 , 12/200 , 0 , 26/200 , 0 , 12/200 , 0],
            [5/200 , 0 , 12/200 , 0 , 12/200 , 0 , 5/200]]).reshape(4,7)

burkes = np.array([[0 , 0 , 0 , 8/32 , 4/32],
         [2/32 , 4/32 , 8/32 , 4/32 , 2/32]]).reshape(2,5)

sierra = np.array([[0 , 0 , 0 , 5/32 , 3/32],
            [2/32 , 4/32 , 5/32 , 4/32 , 2/32],
            [0 , 2/32 , 3/32 , 2/32 , 0]]).reshape(3, 5)

stucki = np.array([[0 , 0 , 0, 8/32 , 4/42],
            [2/42 , 4/42 , 8/42 , 4/42 , 2/32],
            [1/42 , 2/42 , 4/42 , 2/42 , 1/42]]).reshape(3,5)

jarvis = np.array([[0 , 0 , 0, 7/48 , 5/48],
            [3/48 , 5/48 , 7/48 , 5/48 , 3/48],
            [1/48 , 3/48 , 5/48 , 3/48 , 1/48]]).reshape(3,5)


#Getting all files
def getImages(color, image_dir):
    data_path = os.path.join(image_dir, '*g')
    files = glob.glob(image_dir + "/*.png")
    data=[]
    names = []
    if (color=="colorful"):
        for fl in files:
            img = cv2.imread(fl, cv2.IMREAD_COLOR)
            data.append(img)
            names.append(os.path.splitext(os.path.basename(fl))[0])
    elif(color == "gray"):
        for fl in files:
            img = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
            data.append(img)
            names.append(os.path.splitext(os.path.basename(fl))[0])
    else:
        print("please specify color type")
    return data, names

def goThroughGray(matrix, bordersize, filtro):
    startOfFilter = math.floor(filtro.shape[1]/2)

    (row,column) = matrix.shape
    print(row, column)
    #matrixAux = np.copy(matrix)
    matrixAux = matrix.copy()
    row-= bordersize
    column-=bordersize

    interval = [column, bordersize]


    # for i in range(bordersize, row):
    #     for j in range(bordersize, column):
    #         print(matrix[i,j])

    one = 1
    #origem fica em cima
    for i in range(bordersize, row):#i row, j column
    #for i in range(0, row):#i row, j column
        #if(i%2!=0):
            # for j in reversed(range(bordersize, column)):
        temp = interval[0]
        interval[0] = interval[1]
        interval[1] = temp
        one = -one
        #print(i, interval[1], interval[0], one)
        for j in range(interval[1], interval[0], one):
        #for j in range(column-1, -1, -1):
            #print(j)
            if(matrix[i,j] < 128):
                matrixAux[i,j] = 0
            else:
                matrixAux[i,j] = 255
            err = float(matrix[i,j] - matrixAux[i,j])
            #print(err)
            #print(matrix[i,j], matrixAux[i,j], err)
            #if(i>0 and j-1<column-1):
            # matrix[i,j+one] += (7/16)*err
            # #if(i+1<row-1):
            #     #if(j+1<column-1):
            # matrix[i+1,j-one] += (3/16)*err
            #     #if(j<column-1):
            # matrix[i+1, j] += (5/16)*err
            #     #if(j-1<column-1):
            # matrix[i+1,j+one] += (1/16)*err

            for g in range(0, filtro.shape[0]):
                for k in range(0, filtro.shape[1]):
                    matrix[i+g, j+((k-startOfFilter)*one)] += filtro[g,k] * err
            
        # else:
        #     for j in range(bordersize, column):
        #     #for j in range(0, column):
        #         #print(j)
        #         if(matrix[i,j] < 128):
        #             matrixAux[i,j] = 0
        #         else:
        #             matrixAux[i,j] = 255
        #         err = float(matrix[i,j] - matrixAux[i,j])
        #         #print(err)
        #         #if(i>0 and j+1<column-1):
        #         matrix[i, j+1] += (7/16)*err
        #         #if(i+1<row-1):
        #             #if(j-1<column-1):
        #         matrix[i+1,j-1] += (3/16)*err
        #             #if(j<column-1):
        #         matrix[i+1, j] += (5/16)*err
        #             #if(j+1<column-1):
        #         matrix[i+1,j+1] += (1/16)*err


    # plt.axis("off")
    # plt.imshow(matrixAux, cmap="gray", vmin=0, vmax=255)

    # plt.waitforbuttonpress(0)
    # plt.close("all")

    
    # for i in range(0, matrixAux.shape[0]):
    #     for j in range(0, matrixAux.shape[1]):
    #         if(matrixAux[i,j] != 0 and matrixAux[i,j] != 255):
    #             print(matrix[i,j])

    # for i in range(bordersize, row):
    #     for j in range(bordersize, column):
    #         print(matrixAux[i,j])


    return matrixAux



def testColor(title, matrixColor, filtro):
   
    #matrixColor = cv2.cvtColor(cv2.imread("train.png"), cv2.COLOR_BGR2RGB)
   
    #matrixColor = cv2.cvtColor(matrixColor, cv2.COLOR_BGR2RGB)
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
    matrixR = goThroughGray(matrixR, bordersize, filtro)
    matrixG = goThroughGray(matrixG, bordersize, filtro)
    matrixB = goThroughGray(matrixB, bordersize, filtro)


   
    
    matrixColor = cv2.merge((matrixB, matrixG, matrixR))
    final = matrixColor[bordersize:matrixColor.shape[1]-bordersize,bordersize:matrixColor.shape[0]-bordersize].copy()
    # plt.title("color")
    # plt.axis("off")
    # plt.imshow(final)
    # plt.waitforbuttonpress(0)
    # plt.close("all")
    cv2.imwrite(title + 'ColorFloydSteinberg.png', cv2.bitwise_not(final))


def testGray(title, matrixGray, filtro):
    #goThroughGray()
    #matrixGray = cv2.cvtColor(cv2.imread("baboon.png"), cv2.COLOR_BGR2GRAY)
    #matrixColor = cv2.imread('baboon.png', cv2.IMREAD_COLOR)
    
    
    # b, g, r = cv2.split(matrixColor)
    # matrixR = goThroughGray(r, "r")
    # matrixG = goThroughGray(g, "g")
    # matrixB = goThroughGray(b, "b")

    bordersize=5
    matrixGray = cv2.copyMakeBorder(matrixGray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])

    # plt.axis("off")
    # plt.imshow(matrixGray, cmap="gray", vmin=0, vmax=255)
    # plt.waitforbuttonpress(0)
    # plt.close("all")

   
    matrixGray = goThroughGray(matrixGray, bordersize, filtro)

    #cut border out
    matrixFinal = matrixGray[bordersize:matrixGray.shape[1]-bordersize,bordersize:matrixGray.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'GrayFloydSteinberg.png', matrixFinal)

    # plt.axis("off")
    # plt.imshow(matrixFinal, cmap="gray", vmin=0, vmax=255)
    # plt.waitforbuttonpress(0)
    # plt.close("all")
    # for i in range(0, matrixFinal.shape[0]):
    #     for j in range(0, matrixFinal.shape[1]):
    #         if(matrixFinal[i,j] != 0 and matrixFinal[i,j] != 255):
    #             print(matrixFinal[i,j])



if __name__ == '__main__':

    #Color images
    data, names = getImages("colorful", "/home/marcelomartins/Documentos/8sem/imagens/proj1/images")
    for i in range(len(data)):
        testColor(names[i], data[i], floyd)

    #Gray images
    data, names = getImages("gray", "/home/marcelomartins/Documentos/8sem/imagens/proj1/images")
    for i in range(len(data)):
        testGray(names[i], data[i], floyd)
    

    #print(jarvis.shape)

    # img = cv2.cvtColor(cv2.imread("images/baboon.png"), cv2.IMREAD_COLOR)
    # testColor("teste", img, floyd)

    # plt.axis("off")
    # plt.imshow(img)
    # plt.waitforbuttonpress(0)
    # plt.close("all")

    # # matriz = np.arange(40).reshape(5,8)
    # # print(matriz[3, 7], matriz.shape[0])
