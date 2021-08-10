import matplotlib.image as pltimg
import numpy as np
import math
import heapq
import struct

# ************ constants ***********

lq = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

cq = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

zigzag = np.array([
    [0, 1, 5, 6, 14, 15, 27, 28],
    [2, 4, 7, 13, 16, 26, 29, 42],
    [3, 8, 12, 17, 25, 30, 41, 43],
    [9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]
])

Y_DC_huffman_lenCnt = [0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0]
Y_AC_huffman_lenCnt = [0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125]
C_DC_huffman_lenCnt = [0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
C_AC_huffman_lenCnt = [0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119]

# T = np.array([[0.257,0.564,0.098],[-0.148,-0.291,0.439],[0.439,-0.368,-0.071]])
# bias = np.array([16,128,128])
T = np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])
bias = np.array([-128,0,0])
# bias = np.array([0,128,128])

pi = math.acos(-1)
Cos = np.zeros((8,8))
for x in range(8) :
    for u in range(8) :
        Cos[x][u] = math.cos((2*x+1)*u*pi/16)
sqrt2 = math.sqrt(2)

# ************ compression ***********

def scaling_qTable(qf,lq,cq) :
    scalingFactor = (100-qf)/50 if qf>=50 else (50/qf)
    if scalingFactor==0 :
        for i in range(8) :
            for j in range(8) :
                lq[i][j] = 1
                cq[i][j] = 1
    else :
        for i in range(8) :
            for j in range(8) :
                lq[i][j] = round(lq[i][j]*scalingFactor)
                cq[i][j] = round(cq[i][j]*scalingFactor)

def from_RGB_to_YCbCr(img_rgb) :
    img_ycbcr = np.dot(img_rgb,T.T) + bias
    return img_ycbcr

def subsample(img) :
    return img[::2,::2]

def resize_16(img) :
    newImg = np.zeros((((img.shape[0]+15)>>4)<<4, ((img.shape[1]+15)>>4)<<4, img.shape[2]))
    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            newImg[i][j] = img[i][j].copy()
    return newImg

def DCT_and_quantize(img, qTable) :
    F = np.zeros((8,8), dtype=int)
    for u in range(8) :
        for v in range(8) :
            Ff = 0
            for x in range(8) :
                for y in range(8) :
                    Ff += Cos[x][u]*Cos[y][v]*img[x][y]
            Cu = sqrt2/2 if u==0 else 1
            Cv = sqrt2/2 if v==0 else 1
            Ff *= Cu*Cv/4
            F[u][v] = int(round(Ff/qTable[u][v]))
    return F

def get_size(x) :
    x = abs(x)
    size = 0
    while (x>0) :
        x >>= 1
        size += 1
    return size

def run_length(img) :
    zz = [0]*64
    for x in range(8) :
        for y in range(8) :
            zz[zigzag[x][y]] = img[x][y]
    zero = 0
    curAC = []
    for k in range(1,len(zz)) :
        if (zz[k]==0) :
            zero += 1
        else :
            while (zero>15) :
                curAC.append([15<<4,0])
                zero -= 16
            curAC.append([(zero<<4)|get_size(zz[k]),zz[k]])
            zero = 0
    if zero :
        curAC.append([0,0])
    return curAC

def encode_block(img,qTable,allDC,allAC,lastDC) :
    img = DCT_and_quantize(img,qTable)
    dif = img[0][0]-lastDC
    DCpair, ACpairs = [get_size(dif),dif], run_length(img)
    allDC.append(DCpair)
    allAC += ACpairs
    return img[0][0], [DCpair]+ACpairs

def rgb_to_jpeg(img_rgb, fileName) :
    # scaling_qTable(qf,lq,cq)
    img_ycbcr = from_RGB_to_YCbCr(img_rgb)
    img_ycbcr = resize_16(img_ycbcr)
    MCUnum = (img_ycbcr.shape[0]>>4,img_ycbcr.shape[1]>>4)
    img_y, img_cb, img_cr = img_ycbcr[:,:,0], subsample(img_ycbcr[:,:,1]), subsample(img_ycbcr[:,:,2])
    allAC, allDC, MCU = [[],[]], [[],[]], []
    lastYDC, lastCbDC, lastCrDC = 0,0,0
    # for each MCU, the order is Y, Y, Y, Y, Cb, Cr
    for x in range(MCUnum[0]) :
        for y in range(MCUnum[1]) :
            curMCU = []
            for yx in range(2) :
                for yy in range(2) :
                    Y = img_y[(x<<4)+(yx<<3):(x<<4)+(yx<<3)+8, (y<<4)+(yy<<3):(y<<4)+(yy<<3)+8]
                    lastYDC, codes = encode_block(Y,lq,allDC[0],allAC[0],lastYDC)
                    curMCU.append(codes)
            Cb = img_cb[x<<3:(x<<3)+8, y<<3:(y<<3)+8]
            lastCbDC, codes = encode_block(Cb,cq,allDC[1],allAC[1],lastCbDC)
            curMCU.append(codes)
            Cr = img_cr[x<<3:(x<<3)+8, y<<3:(y<<3)+8]
            lastCrDC, codes = encode_block(Cr,cq,allDC[1],allAC[1],lastCrDC)
            curMCU.append(codes)
            MCU.append(curMCU)
    
    # return allDC, allAC, MCU
    write_jpg(fileName,img_rgb.shape,allDC,allAC,MCU)

def calc_frequency(seq, size) :
    fq = [[0,i] for i in range(size)]
    for x in seq :
        fq[x[0]][0] += 1
    if size==256 :
        for i in range(16) :
            for j in range(11,16) :
                fq[(i<<4)|j][0] = -1
            if (i!=0 and i!=15) :
                fq[i<<4][0] = -1
    fq.sort(reverse=True)
    return fq

def calc_coding_table(huffman_lenCnt, seq, size) :
    fq = calc_frequency(seq,size)
    
    lenTable = []
    for i in range(len(huffman_lenCnt)) :
        lenTable += [i]*huffman_lenCnt[i]
    
    curCode, curlen = 0, 0
    codingTable = np.zeros((size,2), dtype=int)
    for i in range(len(lenTable)) :
        while (curlen<lenTable[i]) :
            curlen += 1
            curCode <<= 1
        codingTable[fq[i][1]][0] = curCode
        codingTable[fq[i][1]][1] = curlen
        curCode += 1
    return codingTable

# ************ writing binarny file ***********

def write_APP0() :
    s = b'\x4a\x46\x49\x46\x00\x01\x01\x01\x00\x48\x00\x48\x00\x00'
    size = len(s)+2
    return b'\xff\xe0'+struct.pack(">h",size)+s

def write_DQT(qTable, qId) :
    s = struct.pack("b",qId)
    for i in range(len(qTable)) :
        for j in range(len(qTable)) :
            s += struct.pack("b",qTable[i][j])
    size = len(s)+2
    return b'\xff\xdb'+struct.pack(">h",size)+s

def write_SOF0(shape) :
    s = b'\x08'+struct.pack(">hh",shape[0],shape[1])+b'\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01'
    size = len(s)+2
    return b'\xff\xc0'+struct.pack(">h",size)+s

def write_DHT(huffman_lenCnt, seq, isAC, HTid) :
    s = struct.pack("b",(isAC<<4)|HTid)
    s += bytes(huffman_lenCnt[1:])
    totCnt = sum(huffman_lenCnt)
    fq = calc_frequency(seq, 256 if isAC else 12)
    for i in range(totCnt) :
        s += struct.pack("B",fq[i][1])
    size = len(s)+2
    return b'\xff\xc4'+struct.pack(">h",size)+s

class s_buf() :
    def __init__(self) :
        self.curS = 0
        self.curLen = 0
        self.s = b''
    
    def add_to_buf(self,x,xlen) :
        self.curS = (self.curS<<xlen)|x
        self.curLen += xlen
        while (self.curLen>7) :
            curByte = self.curS>>(self.curLen-8)
            self.s += bytes([curByte])
            if (curByte==255) :
                self.s += bytes([0])
            self.curS -= curByte<<(self.curLen-8)
            self.curLen -= 8
    
    def clear_res(self) :
        self.add_to_buf(0x7f,7)

def add_s(codingTable,x,xlen,buf) :
    buf.add_to_buf(codingTable[x[0]][0],codingTable[x[0]][1])
    buf.add_to_buf(x[1] if x[1]>=0 else (x[1]-1)&((1<<xlen)-1),xlen)

def write_SOS(DC,AC,MCU) :
    s = b'\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00'
    size = len(s)+2
    s = b'\xff\xda'+struct.pack(">h",size)+s
    
    Y_DC_codingTable = calc_coding_table(Y_DC_huffman_lenCnt, DC[0], 12)
    C_DC_codingTable = calc_coding_table(C_DC_huffman_lenCnt, DC[1], 12)
    Y_AC_codingTable = calc_coding_table(Y_AC_huffman_lenCnt, AC[0], 256)
    C_AC_codingTable = calc_coding_table(C_AC_huffman_lenCnt, AC[1], 256)
    buf = s_buf()
    for mcu in MCU :
        for i in range(4) :
            add_s(Y_DC_codingTable,mcu[i][0],mcu[i][0][0],buf)
            for j in range(1,len(mcu[i])) :
                add_s(Y_AC_codingTable,mcu[i][j],mcu[i][j][0]&15,buf)
        for i in range(4,6) :
            add_s(C_DC_codingTable,mcu[i][0],mcu[i][0][0],buf)
            for j in range(1,len(mcu[i])) :
                add_s(C_AC_codingTable,mcu[i][j],mcu[i][j][0]&15,buf)
    buf.clear_res()
    
    return s+buf.s

def write_jpg(fileName,shape,DC,AC,MCU) :
    with open(fileName+"_my.jpg","wb") as f :
        f.write(b'\xff\xd8')
        f.write(write_APP0())
        f.write(write_DQT(lq,0))
        f.write(write_DQT(cq,1))
        f.write(write_SOF0(shape))
        f.write(write_DHT(Y_DC_huffman_lenCnt,DC[0],0,0))
        f.write(write_DHT(Y_AC_huffman_lenCnt,AC[0],1,0))
        f.write(write_DHT(C_DC_huffman_lenCnt,DC[1],0,1))
        f.write(write_DHT(C_AC_huffman_lenCnt,AC[1],1,1))
        f.write(write_SOS(DC,AC,MCU))
        f.write(b'\xff\xd9')