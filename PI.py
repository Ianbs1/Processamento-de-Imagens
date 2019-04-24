

########################################
#
# Nome: Ian Bulhões Santana
# Matricula: 201500018074
# Email:ianbs@dcomp.ufs.br
#
# Nome: Adam Lucas Pinheiro da Silva
# Matricula:201500017836
# Email:adam.silva@dcomp.ufs.br
#
########################################
import numpy as np
import matplotlib.pylab as plt
import math



def imread(imfile): #Q.2
    im = plt.imread(imfile)
    if im.dtype == 'uint8':
        return im
    else:
        return np.uint8(im*255)


    

def nchannels (im): #Q.3
    if len(im.shape) == 3:
        return im.shape[2]
    else:
        return 1

def size (im): #Q.4
    return im.shape[1], im.shape[0]


    
def rgb2gray(im): #Q.5
    imgray = im[:,:,0]*0.299 + im[:,:,1]*0.587 + im[:,:,2]*0.114
    return np.uint8(imgray)


def imreadgray(imgname): #Q.6
     imgray = imread(imgname)
     imgr = rgb2gray(imgray)
     return  plt.imshow(imgr, 'Greys_r')
 
def imshow(im): #Q.7
    if nchannels(im) == 1 :
        img = plt.imshow(im, interpolation = 'nearest')
        img.set_cmap('gray')
        return img
    else:
        img = plt.imshow(im, interpolation = 'nearest')
        return img
        
def thresh(img, limiar): #Q.8
    im1 = rgb2gray(img)
    for x in range(im1.shape[0]):
        for y in range(im1.shape[1]):
            m = im1[x,y]
            if (m>=limiar):
                im1[x,y] = 255
            else:
                im1[x,y] = 0
    return im1
            
    
    

def negative(img): #Q.9
    if nchannels(img) == 1:
        return np.array(255 - img[:], dtype = "uint8")
    else:
        return np.array(255 - img[:,:,:3], dtype = "uint8")



def contrast(img, r, m): #Q.10
    g = np.array(img, dtype="float64")
    g = r * (g - m) + m
    g = g.clip(0, 255)
    g = g.astype(np.uint8)
    return g


def hist (img):
    if nchannels(img) == 1:
        mat = np.empty(256, dtype = "int64")
        for i in range(0, 256):
            mat[i] = np.count_nonzero(img == i)
        return mat
    else:
        mat = np.empty((3,256), dtype = "int64")
        for i in range(0, 256):
            mat[0][i] = np.count_nonzero(img[:,:,0] == i)
            mat[1][i] = np.count_nonzero(img[:,:,1] == i)
            mat[2][i] = np.count_nonzero(img[:,:,2] == i)
        return np.transpose(mat)

def showhist(hist,bi): #Q.12 #Q.13
	hist=np.transpose(hist)
	if hist.shape == (256,):
		hist1= np.arange(1)		
		if bi > 1:
			col = math.ceil(256/bi)
			hist1 = np.zeros(col, dtype="int64")

			cont = 0
			soma = 0
			index = 0
			for j in range(0, 256):
				soma += hist[j]
				cont += 1
				if(cont % bi == 0 or j == 255):
					hist1[index] = soma
					index += 1
					soma = 0
		else:
			hist1=hist



		
		x = np.arange(0,math.ceil(256/bi))
		x1 = plt.subplot(111)
		
		x1.bar(x,hist1,width=0.2,color='gray',align='center')
		
		x1.autoscale(tight=True)		
		plt.show()

	else:
		hist1= np.arange(1)		
		if bi > 1:
			col = math.ceil(256/bi)
			hist1 = np.zeros((3,col), dtype="int64")

			for i in range (0, 3):
				cont = 0
				soma = 0
				index = 0
				for j in range(0, 256):
					soma += hist[i,j]
					cont += 1
					if(cont % bi == 0 or j == 255):
						hist1[i,index] = soma
						index += 1
						soma = 0
		else:
			hist1=hist



		
		x=np.arange(0,math.ceil(256/bi))
		x1 = plt.subplot(111)
		x1.bar(x-0.2,hist1[0],width=0.2,color='r',align='center')
		x1.bar(x,hist1[1],width=0.2,color='g',align='center')
		x1.bar(x+0.2,hist1[2],width=0.2,color='b',align='center')
		x1.autoscale(tight=True)		
		plt.show()
        
        

def histeq(img): #Q.14
	h = np.transpose(hist(img))
	N = img.shape
	equalized_image = np.zeros_like(img)
	equalized_image.astype(np.float32)
	
	if nchannels(img) == 1:#GRAYSCALE
		cdf = np.array([sum(h[:i+1]) for i in range(len(h))]) #calculando histograma acumulado
		equalized_image = ((cdf[img[:,:]] - min(cdf))/((N[0]*N[1])-1))*255
	else:#rgb
		cdf_r = np.array([sum(h[0,:i+1]) for i in range(len(h[0]))]) #calculando histograma acumulado de red
		cdf_g = np.array([sum(h[1,:i+1]) for i in range(len(h[1]))]) #calculando histograma acumulado de green
		cdf_b = np.array([sum(h[2,:i+1]) for i in range(len(h[2]))]) #calculando histograma acumulado de blue
		
		equalized_image[:,:,0] = ((cdf_r[img[:,:,0]] - min(cdf_r))/((N[0]*N[1])-1))*255
		equalized_image[:,:,1] = ((cdf_g[img[:,:,1]] - min(cdf_g))/((N[0]*N[1])-1))*255
		equalized_image[:,:,2] = ((cdf_b[img[:,:,2]] - min(cdf_b))/((N[0]*N[1])-1))*255
	
	equalized_image = equalized_image.round()
	return np.array(equalized_image,dtype='uint8')


def maskBlur():#Q.16
    return 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])



def convolve(img, mask): #Q.15
    x1 = int(mask.shape[0] / 2)
    extra_y = int(mask.shape[1] / 2)

    max_x = img.shape[0]
    max_y = img.shape[1]
    
    c_image = np.copy(img)

    cinza = nchannels(img) == 1

    if cinza:
        aux_image = np.zeros((max_x + x1*2, max_y + extra_y*2), dtype="uint8")
        aux_image[x1:max_x + x1, extra_y:max_y + extra_y] = c_image[:,:]

        if x1 > 0:
            aux_image[:x1, :] = aux_image[x1, :]
            aux_image[max_x + x1:max_x + x1 * 2, :] = aux_image[max_x + x1 - 1, :]
            
        if extra_y > 0:
            for i in range(extra_y - 1, -1, -1):
                aux_image[:, i] = aux_image[:, i+1]
            for i in range(max_y + extra_y, max_y + extra_y*2):
                aux_image[:, i] = aux_image[:, i-1]
                
        for i in range(x1, max_x + x1):
            for j in range(extra_y, max_y + extra_y):
                valor = np.sum(aux_image[i - x1: i + x1 + 1, j - extra_y: j + extra_y + 1] * mask[:,:])
                valor = abs(valor)
                valor = min(valor, 255)
                c_image[i - x1, j - extra_y] = int(valor)
                

        return c_image
    else:
        aux_image = np.zeros((max_x + x1*2, max_y + extra_y*2, 3), dtype="uint8")
        aux_image[x1:max_x + x1, extra_y:max_y + extra_y, :] = c_image[:,:,:3]

        if x1 > 0:
            aux_image[:x1, :, :] = aux_image[x1, :, :]
            aux_image[max_x + x1:max_x + x1 * 2, :, :] = aux_image[max_x + x1 - 1, :, :]
            
        if extra_y > 0:
            for i in range(extra_y - 1, -1, -1):
                aux_image[:, i, :] = aux_image[:, i+1, :]
            for i in range(max_y + extra_y, max_y + extra_y*2):
                aux_image[:, i, :] = aux_image[:, i-1, :]
                
        for i in range(x1, max_x + x1):
            for j in range(extra_y, max_y + extra_y):
                valor0 = np.sum(aux_image[i - x1: i + x1 + 1, j - extra_y: j + extra_y + 1, 0] * mask[:,:])
                valor1 = np.sum(aux_image[i - x1: i + x1 + 1, j - extra_y: j + extra_y + 1, 1] * mask[:,:])
                valor2 = np.sum(aux_image[i - x1: i + x1 + 1, j - extra_y: j + extra_y + 1, 2] * mask[:,:])
                valor0 = abs(valor0)
                valor0 = min(valor0, 255)
                valor1 = abs(valor1)
                valor1 = min(valor1, 255)
                valor2 = abs(valor2)
                valor2 = min(valor2, 255)
                c_image[i - x1, j - extra_y, 0] = int(valor0)
                c_image[i - x1, j - extra_y, 1] = int(valor1)
                c_image[i - x1, j - extra_y, 2] = int(valor2)

        return c_image
    
def blur(img):#Q.17
    return convolve(img, maskBlur())



def seSquare3():#Q.18
    return np.array([[1,1,1],[1,1,1],[1,1,1]], dtype="uint8")

def seCross3():#Q.19
    return np.array([[0,1,0],[1,1,1],[0,1,0]], dtype="uint8")

def erode(img, se):#Q.20
    extra_x = int(se.shape[0] / 2)
    extra_y = int(se.shape[1] / 2)

    max_x = img.shape[0]
    max_y = img.shape[1]
    
    c_image = np.copy(img)

    cinza = nchannels(img) == 1

    if cinza:
        aux_image = np.zeros((max_x + extra_x*2, max_y + extra_y*2), dtype="uint8")
        aux_image[extra_x:max_x + extra_x, extra_y:max_y + extra_y] = c_image[:,:]

        if extra_x > 0:
            aux_image[:extra_x, :] = aux_image[extra_x, :]
            aux_image[max_x + extra_x:max_x + extra_x * 2, :] = aux_image[max_x + extra_x - 1, :]
            
        if extra_y > 0:
            for i in range(extra_y - 1, -1, -1):
                aux_image[:, i] = aux_image[:, i+1]
            for i in range(max_y + extra_y, max_y + extra_y*2):
                aux_image[:, i] = aux_image[:, i-1]
                
        for i in range(extra_x, max_x + extra_x):
            for j in range(extra_y, max_y + extra_y):
                valor = np.max(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1] * se[:,:])
                c_image[i - extra_x, j - extra_y] = int(valor)
                

        return c_image
    else:
        aux_image = np.zeros((max_x + extra_x*2, max_y + extra_y*2, 3), dtype="uint8")
        aux_image[extra_x:max_x + extra_x, extra_y:max_y + extra_y, :] = c_image[:,:,:3]

        if extra_x > 0:
            aux_image[:extra_x, :, :] = aux_image[extra_x, :, :]
            aux_image[max_x + extra_x:max_x + extra_x * 2, :, :] = aux_image[max_x + extra_x - 1, :, :]
            
        if extra_y > 0:
            for i in range(extra_y - 1, -1, -1):
                aux_image[:, i, :] = aux_image[:, i+1, :]
            for i in range(max_y + extra_y, max_y + extra_y*2):
                aux_image[:, i, :] = aux_image[:, i-1, :]
                
        for i in range(extra_x, max_x + extra_x):
            for j in range(extra_y, max_y + extra_y):
                valor0 = np.max(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1, 0] * se[:,:])
                valor1 = np.max(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1, 1] * se[:,:])
                valor2 = np.max(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1, 2] * se[:,:])
                c_image[i - extra_x, j - extra_y, 0] = int(valor0)
                c_image[i - extra_x, j - extra_y, 1] = int(valor1)
                c_image[i - extra_x, j - extra_y, 2] = int(valor2)

        return c_image


def dilate(img, se):#Q.21
    extra_x = int(se.shape[0] / 2)
    extra_y = int(se.shape[1] / 2)

    max_x = img.shape[0]
    max_y = img.shape[1]
    
    c_image = np.copy(img)
    se_aux = np.array(se * (-254) + 255, dtype="int64")
    cinza = nchannels(img) == 1

    if cinza:
        aux_image = np.zeros((max_x + extra_x*2, max_y + extra_y*2), dtype="uint8")
        aux_image[extra_x:max_x + extra_x, extra_y:max_y + extra_y] = c_image[:,:]

        if extra_x > 0:
            aux_image[:extra_x, :] = aux_image[extra_x, :]
            aux_image[max_x + extra_x:max_x + extra_x * 2, :] = aux_image[max_x + extra_x - 1, :]
            
        if extra_y > 0:
            for i in range(extra_y - 1, -1, -1):
                aux_image[:, i] = aux_image[:, i+1]
            for i in range(max_y + extra_y, max_y + extra_y*2):
                aux_image[:, i] = aux_image[:, i-1]
                
        for i in range(extra_x, max_x + extra_x):
            for j in range(extra_y, max_y + extra_y):
                valor = np.min(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1] * se_aux[:,:])
                
                c_image[i - extra_x, j - extra_y] = int(valor)
                

        return c_image
    else:
        aux_image = np.zeros((max_x + extra_x*2, max_y + extra_y*2, 3), dtype="uint8")
        aux_image[extra_x:max_x + extra_x, extra_y:max_y + extra_y, :] = c_image[:,:,:3]

        if extra_x > 0:
            aux_image[:extra_x, :, :] = aux_image[extra_x, :, :]
            aux_image[max_x + extra_x:max_x + extra_x * 2, :, :] = aux_image[max_x + extra_x - 1, :, :]
            
        if extra_y > 0:
            for i in range(extra_y - 1, -1, -1):
                aux_image[:, i, :] = aux_image[:, i+1, :]
            for i in range(max_y + extra_y, max_y + extra_y*2):
                aux_image[:, i, :] = aux_image[:, i-1, :]
                
        for i in range(extra_x, max_x + extra_x):
            for j in range(extra_y, max_y + extra_y):
                valor0 = np.min(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1, 0] * se_aux[:,:])
                valor1 = np.min(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1, 1] * se_aux[:,:])
                valor2 = np.min(aux_image[i - extra_x: i + extra_x + 1, j - extra_y: j + extra_y + 1, 2] * se_aux[:,:])
                c_image[i - extra_x, j - extra_y, 0] = int(valor0)
                c_image[i - extra_x, j - extra_y, 1] = int(valor1)
                c_image[i - extra_x, j - extra_y, 2] = int(valor2)

        return c_image

