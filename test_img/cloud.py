#绘制disp_x.csv和disp_y.csv的云图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
from PIL import Image
import os

#两个都是二位矩阵
disp_x = pd.read_csv('_disp_x.csv',header=None).values
disp_y = pd.read_csv('_disp_y.csv',header=None).values

# print(disp_x)
# print(disp_y)

#绘制云图
plt.figure(figsize=(10,10))
plt.imshow(disp_y, cmap='jet', interpolation='nearest')
#显示颜色条
plt.colorbar()
plt.axis('off')
plt.show()


#最左边的空间周期为10，最右边的空间周期为150