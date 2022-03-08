import cv2
import os

def grary_to_binary(src, threshold, aim_color,path):
    img = cv2.imread(src)
    ret, dst = cv2.threshold(img, threshold, aim_color, cv2.THRESH_BINARY)
    cv2.imwrite(path, dst)


if __name__ == '__main__':
    label_path = "../data_set/medical_data_set/labels/"     #  标签路径
    out_path = "../data_set/medical_data_set/labels/transform/"     #  输出路径
    threshold = 16
    '''
    for i in range(520):
        src = label_path + str(i) + '.png'
        out = out_path + str(i) + '.png'
        grary_to_binary(src, threshold, 1, out)
    '''
    file = open("../data_set/medical_data_set/train_val.txt", 'w')
    for i in range(416, 520):
        file.write(str(i)+'\n')
    file.close()