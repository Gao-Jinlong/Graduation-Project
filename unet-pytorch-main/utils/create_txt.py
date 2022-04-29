import os
from tqdm import tqdm

if __name__ == '__main__':
    data_root = 'E:/00_graduation project/DataSet/isic2017'
    data_type = 'ISBI2016_ISIC_Part1_Test_Data'

    list = os.listdir(os.path.join(data_root, data_type))
    txt = open(os.path.join(data_root, 'val.txt'), 'w')
    for i in tqdm(list):
        i = i.split('.')[0]  # 删掉后缀
        txt.write(i + '\n')
