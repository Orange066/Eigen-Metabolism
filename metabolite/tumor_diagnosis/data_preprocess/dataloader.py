from pyimzML.pyimzml.ImzMLParser import ImzMLParser
import random
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
normal_files = ['YQF-N-T/YQF-pos-normal_mucosa-1.txt']
tumor_files = ['YQF-N-T/YQF-pos-tumor-1.txt','YQF-N-T/YQF-pos-tumor-2.txt','YQF-N-T/YQF-pos-tumor-3.txt']
p = ImzMLParser("YQF-pos.imzML")
mzArray, intensity = p.getspectrum(38 * 48)
print(intensity)
print(intensity.shape)
def extractAxis(file):
    all_index = []
    count = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split( )
            if(line[3] == "1"):
                index = int(line[1]) * int(line[2])
                all_index.append(index)
                count += 1
        # print(all_index)
    f.close()
    return all_index, count

# extractFeature('YQF-N-T/YQF-pos-tumor-1.txt')
def countAll(imzML,indexs,count):
    p = ImzMLParser(imzML)
    allInten = np.zeros(914)
    for index in indexs:
        mzArray, intensity = p.getspectrum(index)
        allInten += intensity
    allInten /= count
    return allInten,mzArray


def save_to_csv(indexes,imzml):
    ndarry = []
    p = ImzMLParser(imzml)
    for index in indexes:
        mzArray, intensity = p.getspectrum(38 * 48)
        print(intensity)
        # print(intensity.shape)
        ndarry.append(intensity)
    # np.save('data/normal.npy',ndarry)
    

def countAllNorm(imzML,indexs):
    p = ImzMLParser(imzML)
    allInten = np.zeros(914)
    for index in indexs:
        mzArray, intensity = p.getspectrum(index)
        allInten += intensity
    _range = np.max(allInten) - np.min(allInten)
    allInten = (allInten - np.min(allInten)) / _range

    return allInten,mzArray

# def draw(mzArray,result,threshold,filename):
#     plt.plot(mzArray,result)
#     for a, b in zip(mzArray, result):
#         if (b > 4 or b == -1):
#             plt.text(a, b, a, ha='center', va='bottom',size=6)
#     plt.savefig(filename)
#     plt.close()

def chu(result1,result2):
    result = np.ones(914)
    for i in range(len(result1)):
        if result2[i] == 0 and result1[i] != 0:
            result[i] = -1 
        if result2[i] != 0:
            result[i] = result1[i] / result2[i]
    return result

# indexes, count = extractAxis('YQF-N-T/YQF-pos-normal_mucosa-1.txt')
# save_to_csv(indexes,"YQF-pos.imzML")




# indexs1, count1 = extractFeature('YQF-N-T/YQF-pos-normal_mucosa-1.txt')
# indexs2, count2 = extractFeature('YQF-N-T/YQF-pos-tumor-1.txt')
# indexs3, count3 = extractFeature('YQF-N-T/YQF-pos-tumor-2.txt')
# indexs4, count4 = extractFeature('YQF-N-T/YQF-pos-tumor-3.txt')
# result1,mzArray = countAll("YQF-pos.imzML",indexs1,count1)
# result2,mzArray = countAll("YQF-pos.imzML",indexs2,count2)
# result3,mzArray = countAll("YQF-pos.imzML",indexs3,count3)
# result4,mzArray = countAll("YQF-pos.imzML",indexs4,count4)

# # draw(mzArray,result1,70,"test_normal.jpg")
# draw(mzArray,chu(result2,result1),2,"testAfter-1.jpg")
# draw(mzArray,chu(result3,result1),2,"testAfter-2.jpg")
# draw(mzArray,chu(result4,result1),2,"testAfter-3.jpg")



# print(mzArray)
