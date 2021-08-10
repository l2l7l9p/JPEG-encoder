import matplotlib.image as pltimg
import jpeg_encoder as jpgenc
import sys

def rgb_to_gif(img_rgb,fileName) :
    pltimg.imsave(fileName+"_my.gif",img_rgb)

def calc_loss(img1,img2) :
    Sum = 0
    for i in range(img1.shape[0]) :
        for j in range(img1.shape[1]) :
            for k in range(img1.shape[2]) :
                Sum += abs(float(img1[i][j][k])-float(img2[i][j][k]))
    Avg = Sum / (img1.shape[0]*img1.shape[1]*img1.shape[2])
    print("sum =",Sum,"avg =",Avg)

def main() :
    fileName = sys.argv[1]
    img_rgb_org = pltimg.imread(fileName+".jpg")
    jpgenc.rgb_to_jpeg(img_rgb_org,fileName)
    rgb_to_gif(img_rgb_org,fileName)
    
    img_rgb_fromJpeg = pltimg.imread(fileName+"_my.jpg")
    img_rgb_fromGif = pltimg.imread(fileName+"_my.gif")
    calc_loss(img_rgb_org,img_rgb_fromJpeg)
    calc_loss(img_rgb_org,img_rgb_fromGif)

if __name__ == "__main__" :
    main()