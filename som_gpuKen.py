from __future__ import division

## Self-organizing map using scipy

## This code uses a square grid rather than hexagonal grid, as scipy allows for fast square grid computation.
from sklearn.preprocessing import normalize
import random
from math import *
import sys
import scipy
import numpy as np
import pandas as pd
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver
import skcuda.linalg as linalg
import skcuda.misc as misc
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
linalg.init()

mod = SourceModule("""
    #include<math.h>
    #include<stdio.h>
    __global__ void fn(float *bmu_vec, float dist, int *min_x, int *min_y, int *max_x, int *max_y)
    {
        int tx = threadIdx.x;
        int bmu_row = (((int)bmu_vec[tx]) / 30);
        int bmu_col = (((int)bmu_vec[tx]) % 30);

        int min_y = 0;
        if(bmu_row - (int)dist > 0){
            min_y = (bmu_row) - (int)(dist);
        }
        int max_y = 30;
        if(bmu_row + (int)dist < 30){
            max_y = bmu_row + (int)dist;
        }
        int min_x = 0;
        if(bmu_col - (int)dist > 0){
            min_x = bmu_col - (int)dist;
        }
        int max_x = 30;
        if(bmu_col + (int)dist < 30){
            max_x = bmu_col + (int)dist;
        }
        
    }
""")

mod2 = SourceModule("""
    #include<math.h>
    #include<stdio.h>
    __global__ void set(float *nodes, float *d_nodes)
    { 
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int idx = col + row * 28;
        if((col >= 28) || (row >= 900))
            return;
        nodes[idx] = nodes[idx] + d_nodes[idx];
    }
""")

class SOM:

    def __init__(self, nodes, height, width, FV_size, learning_rate=0.005):
        self.height = height
        self.width = width
        self.FV_size = FV_size
        self.radius = (height+width)/2
        self.learning_rate = learning_rate
        self.nodes = scipy.array(nodes)
        self.nodes_gpu = gpuarray.to_gpu(scipy.array(nodes))
        self.d_nodes = gpuarray.to_gpu(scipy.array([0 for i in range(self.FV_size*self.width * self.height)]))

    # train_vector: [ FV0, FV1, FV2, ...] -> [ [...], [...], [...], ...]
    # train vector may be a list, will be converted to a list of scipy arrays
    def train(self, iterations=512, train_vector=[[]]): 
        for t in range(len(train_vector)):
            train_vector[t] = scipy.array(train_vector[t])
        time_constant = iterations/log(self.radius)
        for i in range(1, iterations+1):
            self.d_nodes.fill(0)
            alpha = 0.2 - i * 0.01
            numBach = 1
            for batch in range(numBach):
                batch_vector = train_vector[(batch * len(train_vector) / numBach):((batch + 1) * len(train_vector) / numBach)]
                batch_gpu = gpuarray.to_gpu(batch_vector)
                dist_matrix = self.compute_batch(batch_gpu)
                bmu_vec = misc.argmax(dist_matrix,0)
                radius_decaying = np.float32(self.radius * exp(-1.0 * (i+batch) / time_constant))
                rad_div_val = np.float32(2 * radius_decaying * (i+batch))
                learning_rate_decaying = np.float32(self.learning_rate * exp(-1.0 * (i+batch) / time_constant)) 
                func = mod.get_function("fn")
                min_x=0;
                min_y=0;
                max_x=30;
                max_y=30;
                func(bmu_vec.gpudata, radius_decaying, &min_x, &min_y, &max_x, &max_y, block = (1024, 1, 1))

                for j in range(min_x,max_x+1):
                    for k in range(min_y, max_y+1):
                        singleIndex = j+ self.width * k
                        #index_gpu = gpuarray.to_gpu(np.array(range(self.FV_size*singleIndex,self.FV_size*(1+singleIndex))))
                        for l in range(len(train_vector)/numBach):
                            singleDistIndex = misc.get_by_index(bmu_vec, np.array([l])) * numBach + l
                            closestDist = misc.get_by_index(dist_matrix, singleDistIndex) * alpha
                            startInd = misc.get_by_index(bmu_vec, np.array([l])).get()

                            updatePart = misc.get_by_index(self.d_nodes, np.array(range(startInd*self.FV_size,(startInd+1)*self.FV_size)))

                            scr_gpu = 
                            misc.set_by_index(self.d_nodes, index_gpu,) 


            func2 = mod2.get_function("set") 
            tx = 32
            ty = 32
            bx = int((900 + tx - 1)/tx)
            by = int((900 + ty - 1)/ty)
            func2(self.nodes_gpu.gpudata, self.d_nodes.gpudata, block = (tx,ty,1), grid = (bx,by))
    
    def compute_batch(self, batch_gpu):
        bt_gpu = linalg.transpose(batch_gpu)
        distance_matrix_gpu = linalg.dot(self.nodes_gpu, bt_gpu)
        return distance_matrix_gpu

def loadDataset(filename):
    data = pd.read_csv(filename, header=None)
    return data
    
if __name__ == "__main__":
    print("Initialization...") 
    data = np.array(loadDataset("/Shared/bdagroup5/100_HIGGS.csv"))
    newdata = data[:,1:]
    norm = normalize(newdata) 
    nodes = []
    width =  30
    height = 30
    for i in range(width * height):
        ri = random.randint(1, len(norm) - 1)
        nodes.append(norm[ri])
    som = SOM(nodes,width,height,28,0.005)  
    print(som.nodes[0])
    som.train(1000, norm)
    lepton_pt = []
    lepton_eta = []
    lepton_phi = []
    mem = []
    mep = []
    jopt = []
    joeta = []
    jophi = []
    jobtag = []
    jtpg = []
    jteta = []
    jtphi = []
    jtbtag = []
    jthpt = []
    jtheta = []
    jthphi = []
    jthbtag = []
    jfpt = []
    jfeta = []
    jfphi = []
    jfbtag = []
    m_jj = []
    m_jjj = []
    m_lv = []
    m_jlv = []
    m_bb = []
    m_wbb = []
    m_wwbb = []
    som.nodes = som.nodes_gpu.get()
    print(len(som.nodes))
    print(som.nodes.shape)
    for i in range(len(som.nodes)):
        lepton_pt.append(som.nodes[i][0])
        lepton_eta.append(som.nodes[i][1])
        lepton_phi.append(som.nodes[i][2])
        mem.append(som.nodes[i][3])
        mep.append(som.nodes[i][4])
        jopt.append(som.nodes[i][5])
        joeta.append(som.nodes[i][6])
        jophi.append(som.nodes[i][7])
        jobtag.append(som.nodes[i][8])
        jtpg.append(som.nodes[i][9])
        jteta.append(som.nodes[i][10])
        jtphi.append(som.nodes[i][11])
        jtbtag.append(som.nodes[i][12])
        jthpt.append(som.nodes[i][13])
        jtheta.append(som.nodes[i][14])
        jthphi.append(som.nodes[i][15])
        jthbtag.append(som.nodes[i][16])
        jfpt.append(som.nodes[i][17])
        jfeta.append(som.nodes[i][18])
        jfphi.append(som.nodes[i][19])
        jfbtag.append(som.nodes[i][20])
        m_jj.append(som.nodes[i][21])
        m_jjj.append(som.nodes[i][22])
        m_lv.append(som.nodes[i][23])
        m_jlv.append(som.nodes[i][24])
        m_bb.append(som.nodes[i][25])
        m_wbb.append(som.nodes[i][26])
        m_wwbb.append(som.nodes[i][27])
    min1 = np.min(lepton_pt)
    min2 = np.min(lepton_eta)
    min3 = np.min(lepton_phi)
    min4 = np.min(mem)
    min5 = np.min(mep)
    min6 = np.min(jopt)
    min7 = np.min(joeta)
    min8 = np.min(jophi)
    min9 = np.min(jobtag)
    min10 = np.min(jtpg)
    min11 = np.min(jteta)
    min12 = np.min(jtphi)
    min13 = np.min(jtbtag)
    min14 = np.min(jthpt)
    min15 = np.min(jtheta)
    min16 = np.min(jthphi)
    min17 = np.min(jthbtag)
    min18 = np.min(jfpt)
    min19 = np.min(jfeta)
    min20 = np.min(jfphi)
    min21 = np.min(jfbtag)
    min22 = np.min(m_jj)
    min23 = np.min(m_jjj)
    min24 = np.min(m_lv)
    min25 = np.min(m_jlv)
    min26 = np.min(m_bb)
    min27 = np.min(m_wbb)
    min28 = np.min(m_wwbb)
    max1 = np.max(lepton_pt)
    max1 = np.max(lepton_pt)
    max2 = np.max(lepton_eta)
    max3 = np.max(lepton_phi)
    max4 = np.max(mem)
    max5 = np.max(mep)
    max6 = np.max(jopt)
    max7 = np.max(joeta)
    max8 = np.max(jophi)
    max9 = np.max(jobtag)
    max10 = np.max(jtpg)
    max11 = np.max(jteta)
    max12 = np.max(jtphi)
    max13 = np.max(jtbtag)
    max14 = np.max(jthpt)
    max15 = np.max(jtheta)
    max16 = np.max(jthphi)
    max17 = np.max(jthbtag)
    max18 = np.max(jfpt)
    max19 = np.max(jfeta)
    max20 = np.max(jfphi)
    max21 = np.max(jfbtag)
    max22 = np.max(m_jj)
    max23 = np.max(m_jjj)
    max24 = np.max(m_lv)
    max25 = np.max(m_jlv)
    max26 = np.max(m_bb)
    max27 = np.max(m_wbb)
    max28 = np.max(m_wwbb)
    try: 
        from PIL import Image
        print("Saving Image: sompy_test_colors.png...")
        ilepton_pt = Image.new("RGB", (width, height))
        ilepton_eta = Image.new("RGB", (width, height))
        ilepton_phi = Image.new("RGB", (width, height))
        imem = Image.new("RGB", (width, height))
        imep = Image.new("RGB", (width, height))
        ijopt  = Image.new("RGB", (width, height))
        ijoeta = Image.new("RGB", (width, height))
        ijophi = Image.new("RGB", (width, height))
        ijobtag = Image.new("RGB", (width, height))
        ijtpg = Image.new("RGB", (width, height))
        ijteta = Image.new("RGB", (width, height))
        ijtphi = Image.new("RGB", (width, height))
        ijtbtag = Image.new("RGB", (width, height))
        ijthpt = Image.new("RGB", (width, height))
        ijtheta = Image.new("RGB", (width, height))
        ijthphi = Image.new("RGB", (width, height))
        ijthbtag = Image.new("RGB", (width, height))
        ijfpt = Image.new("RGB", (width, height))
        ijfeta = Image.new("RGB", (width, height))
        ijfphi = Image.new("RGB", (width, height))
        ijfbtag = Image.new("RGB", (width, height))
        im_jj = Image.new("RGB", (width, height))
        im_jjj = Image.new("RGB", (width, height))
        im_lv = Image.new("RGB", (width, height))
        im_jlv = Image.new("RGB", (width, height))
        im_bb = Image.new("RGB", (width, height))
        im_wbb = Image.new("RGB", (width, height))
        im_wwbb = Image.new("RGB", (width, height))
        for loc in range(width * height):
            r = int(loc / width)
            c = int(loc % width)
            x = 255*((lepton_pt[loc]-min1)/(max1-min1))
            ilepton_pt.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((lepton_eta[loc]-min2)/(max2-min2))
            ilepton_eta.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((lepton_phi[loc]-min3)/(max3-min3))
            ilepton_phi.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((mem[loc]-min4)/(max4-min4))
            imem.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((mep[loc]-min5)/(max5-min5))
            imep.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jopt[loc]-min6)/(max6-min6))
            ijopt.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((joeta[loc]-min7)/(max7-min7))
            ijoeta.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jophi[loc]-min8)/(max8-min8))
            ijophi.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jobtag[loc]-min9)/(max9-min9))
            ijobtag.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jtpg[loc]-min10)/(max10-min10))
            ijtpg.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jteta[loc]-min11)/(max11-min11))
            ijteta.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jtphi[loc]-min12)/(max12-min12))
            ijtphi.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jtbtag[loc]-min13)/(max13-min13))
            ijtbtag.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jthpt[loc]-min14)/(max14-min14))
            ijthpt.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jtheta[loc]-min15)/(max15-min15))
            ijtheta.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jthphi[loc]-min16)/(max16-min16))
            ijthphi.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jthbtag[loc]-min17)/(max17-min17))
            ijthbtag.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jfpt[loc]-min18)/(max18-min18))
            ijfpt.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jfeta[loc]-min19)/(max19-min19))
            ijfeta.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jfphi[loc]-min20)/(max20-min20))
            ijfphi.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((jfbtag[loc]-min21)/(max21-min21))
            ijfbtag.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((m_jj[loc]-min22)/(max22-min22))
            im_jj.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((m_jjj[loc]-min23)/(max23-min23))
            im_jjj.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((m_lv[loc]-min24)/(max24-min24))
            im_lv.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((m_jlv[loc]-min25)/(max25-min25))
            im_jlv.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((m_bb[loc]-min26)/(max26-min26))
            im_bb.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((m_wbb[loc]-min27)/(max27-min27))
            im_wbb.putpixel((c,r), (int(x), int(x), int(x)))
            x = 255*((m_wwbb[loc]-min28)/(max28-min28))
            im_wwbb.putpixel((c,r), (int(x), int(x), int(x)))
        ilepton_pt = ilepton_pt.resize((width*10, height*10),Image.NEAREST)
        ilepton_eta = ilepton_eta.resize((width*10, height*10),Image.NEAREST)
        ilepton_phi = ilepton_phi.resize((width*10, height*10),Image.NEAREST)
        imem = imem.resize((width*10, height*10),Image.NEAREST)
        imep = imep.resize((width*10, height*10),Image.NEAREST)
        ijopt = ijopt.resize((width*10, height*10),Image.NEAREST)
        ijoeta = ijoeta.resize((width*10, height*10),Image.NEAREST)
        ijophi = ijophi.resize((width*10, height*10),Image.NEAREST)
        ijobtag = ijobtag.resize((width*10, height*10),Image.NEAREST)
        ijtpg = ijtpg.resize((width*10, height*10),Image.NEAREST)
        ijteta = ijteta.resize((width*10, height*10),Image.NEAREST)
        ijtphi = ijtphi.resize((width*10, height*10),Image.NEAREST)
        ijtbtag = ijtbtag.resize((width*10, height*10),Image.NEAREST)
        ijthpt = ijthpt.resize((width*10, height*10),Image.NEAREST)
        ijtheta = ijtheta.resize((width*10, height*10),Image.NEAREST)
        ijthphi = ijthphi.resize((width*10, height*10),Image.NEAREST)
        ijthbtag = ijthbtag.resize((width*10, height*10),Image.NEAREST)
        ijfpt = ijfpt.resize((width*10, height*10),Image.NEAREST)
        ijfeta = ijfeta.resize((width*10, height*10),Image.NEAREST)
        ijfphi = ijfphi.resize((width*10, height*10),Image.NEAREST)
        ijfbtag = ijfbtag.resize((width*10, height*10),Image.NEAREST)
        im_jj = im_jj.resize((width*10, height*10),Image.NEAREST)
        im_jjj = im_jjj.resize((width*10, height*10),Image.NEAREST)
        im_lv = im_lv.resize((width*10, height*10),Image.NEAREST)
        im_jlv = im_jlv.resize((width*10, height*10),Image.NEAREST)
        im_bb = im_bb.resize((width*10, height*10),Image.NEAREST)
        im_wbb = im_wbb.resize((width*10, height*10),Image.NEAREST)
        im_wwbb = im_wwbb.resize((width*10, height*10),Image.NEAREST)
        ilepton_pt.save("1,all.png")
        print("1")
        ilepton_eta.save("2,all.png")
        print("2")
        ilepton_phi.save("3,all.png")
        print("3")
        imem.save("4,all.png")
        print("4")
        imep.save("5,all.png")
        print("5")
        ijopt.save("6,all.png")
        print("6")
        ijoeta.save("7,all.png")
        print("7")
        ijophi.save("8,all.png")
        print("8")
        ijobtag.save("9,all.png")
        print("9")
        ijtpg.save("10,all.png")
        print("10")
        ijteta.save("11,all.png")
        print("11")
        ijtphi.save("12,all.png")
        print("12")
        ijtbtag.save("13,all.png")
        print("13")
        ijthpt.save("14,all.png")
        print("14")
        ijtheta.save("15,all.png")
        print("15")
        ijthphi.save("16,all.png")
        print("16")
        ijthbtag.save("17,all.png")
        print("17")
        ijfpt.save("18,all.png")
        print("18")
        ijfeta.save("19,all.png")
        print("19")
        ijfphi.save("20,all.png")
        print("20")
        ijfbtag.save("21,all.png")
        print("21")
        im_jj.save("22,all.png")
        print("22")
        im_jjj.save("23,all.png")
        print("23")
        im_lv.save("24,all.png")
        print("24")
        im_jlv.save("25,all.png")
        print("25")
        im_bb.save("26,all.png")
        print("26")
        im_wbb.save("27,all.png")
        print("27")
        im_wwbb.save("28,all.png")
        print("28")
    except Exception as err:
        print(err)
        print("Error saving the image, do you have PIL (Python Imaging Library) installed?")
