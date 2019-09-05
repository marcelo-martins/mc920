from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sys
import cv2

# img_src = sys.argv[1]
# option = sys.argv[2]

img = cv2.imread("baboon.png",cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show()
i_max = len(img)
j_max = len(img[0])

halfToning3 = np.array([[6,8,4],
						[1,0,3],
						[5,2,7]])

halfToning4 = np.array([[0,12,3,15],
						[8,4,11,7],
						[2,14,1,13],
						[10,6,9,5]])

new_img = img

option = 3
if (option == 1):
	mask = halfToning3
	maskSize = 3
	rate = 9/255
else:
	mask = halfToning4
	maskSize = 4
	rate = 16/255

if ((option == 1) or (option == 2)):
	#converting to 0-9 interval
	new_img = new_img*rate

	for i in range(i_max):
		for j in range(j_max):
			local_i = i%maskSize
			local_j = j%maskSize
			if (new_img[i][j] < mask[local_i][local_j]):
				new_img[i][j] = 0
			else:
				new_img[i][j] = 255
else:
	for i in range(i_max):
		for j in range(j_max):
			new_i = i
			if (i%2 == 0):
				new_j = j_max-j-1 
			else:
				new_j = j

			pixel = new_img[new_i][new_j]
			if (pixel>128):
				new_img[new_i][new_j] = 255
				error = pixel-255
			else:
				new_img[new_i][new_j] = 0
				error = pixel

			#propagar erro 
			i_aux = new_i
			j_aux = new_j+1	
			if ((i_aux >= 0) and (i_aux < i_max)):
				if ((j_aux >= 0) and (j_aux < j_max)):
					new_img[i_aux][j_aux] += float(error)*(7/16)

			i_aux = new_i+1
			j_aux = new_j-1	
			if ((i_aux >= 0) and (i_aux < i_max)):
				if ((j_aux >= 0) and (j_aux < j_max)):
					new_img[i_aux][j_aux] += float(error)*(3/16)

			i_aux = new_i+1
			j_aux = new_j
			if ((i_aux >= 0) and (i_aux < i_max)):
				if ((j_aux >= 0) and (j_aux < j_max)):
					new_img[i_aux][j_aux] += float(error)*(5/16)

			i_aux = new_i+1
			j_aux = new_j+1
			if ((i_aux >= 0) and (i_aux < i_max)):
				if ((j_aux >= 0) and (j_aux < j_max)):	
					new_img[i_aux][j_aux] += float(error)*(1/16)

plt.imshow(new_img, cmap='gray')
plt.show()
misc.imsave('out.bmp',new_img)