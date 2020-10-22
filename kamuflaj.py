from PIL import Image
import numpy as np
import scipy
import scipy.cluster
import random
import cv2
from matplotlib import pyplot as plt

def openImage(image, width=500, height=500):
    image = Image.open(image)
    image = image.resize((width, height))
    colorArray = createPattern(image)

    gray = image.convert('L')
    bw = np.asarray(gray).copy()
    bw[bw < 128] = 0   
    bw[bw >= 128] = 255 
    bw = Image.fromarray(bw) #Siyah beyaza çevrilmiş numpy tipi image formatına dönüştürüldü.
    pix=bw.load()
    imPix=image.load()
    cBlack = 0
    cWhite = 0
    
    #Resimdeki siyah beyaz sayısının hesaplanması.
    for x in range(height):
        for y in range(width):
            if(pix[x ,y] == 0):
                cBlack = cBlack+1
            else: 
                cWhite = cWhite+1

    #Kamufle olacak nesnenin piksellerinin belirlenmesi ve o konuma göre ana resmin piksellerine renk ataması.
    for x in range(height):
        for y in range(width):
            if cWhite > cBlack:
                if(pix[x ,y] == 0):
                    a = random.choice(colorArray)
                    imPix[x, y] = tuple(a)
            else:
                if(pix[x ,y] == 255):
                    a = random.choice(colorArray)
                    imPix[x, y] = tuple(a)

    #Belirlenen piksel renklerine göre resim oluşturulması.
    img=np.zeros((width,height,3), dtype = "uint8")
    for x in range(height):
        for y in range(width):
            cv2.rectangle(img,(int(y),x),(int(y+1),x+1),imPix[y, x],-1)
    return img

def createPattern(image):
    colorArray = []
    ar = np.asarray(image)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float) 

    codes, dist = scipy.cluster.vq.kmeans(ar, 5) #Baskın 5 rengin kümelenmesi.
    print("codes: ",codes," uzaklık: ",dist)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)
    print("vektör: ",vecs," uzaklik: ", dist)
    counts, bins = scipy.histogram(vecs, len(codes))
    print("counts: ",counts," bins: ", bins)
    index_max = scipy.argmax(counts) #En sık kullanılan 1 rengin indexini bulur
    print("indexmaks: ", index_max)
    peak = codes[index_max]
    print(peak)

    for i in range(0,5):
        colorArray.append(codes[i].astype("uint8").tolist())
        colorArray.append(peak.astype("uint8").tolist()) #En baskın renk
        print("colorArray")
    return colorArray


img = openImage('images/moon.jpg')
imfile = Image.fromarray(img)
imfile.save("images/moonResult.png")
fig = plt.figure()
plt.axis("off")
plt.imshow(img)
fig.patch.set_facecolor('xkcd:black')
plt.savefig('moonResult.png')
plt.show()
