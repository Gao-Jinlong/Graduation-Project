from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# =======================================
# =======================================
obj_class = 1
dataDir='E:/00_graduation project/DataSet/COCO/' # 数据集根目录
dataType='train2017' # 选择图像类型
annFile='{}2017_Train_Val_annotations/instances_{}.json'.format(dataDir,dataType) # annotation路径

# 蒙版绘制函数
def showAnns(mask, anns, c):
    if len(anns) == 0:  # 判断注释类型
        return 0
    if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
        datasetType = 'instances'  # 蒙版
    else:
        raise Exception('datasetType not supported')  # 错误

    if datasetType == 'instances':
        for ann in anns:
            # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0] # 随机生成标记颜色
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:  # polygon多边形
                    for seg in ann['segmentation']:
                        poly = np.array(seg, np.int32).reshape((int(len(seg) / 2), 2))  # 将标记信息转换为坐标数组
                        poly = poly.reshape((-1, 1, 2)) # 修整顶点坐标数组格式
                        cv.fillPoly(mask, [poly], color=c) # 填充形状
        # plt.imshow(mask)
        # plt.show()
        return mask

# 将COCO格式转换为VOC格式
# 图片形状[x,y,3]， COCO标注信息， voc蒙版颜色
def coco2voc(imgShape, anns, color):
    mask = np.zeros(imgShape, np.uint8)  # 绘制纯黑底色

    mask = showAnns(mask, anns, color)  # 绘制mask
    # mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

    # 获取图片ID
    imgId = str(anns[0]['image_id']).zfill(12)
    imgId_ = imgId + '.png'
    src = dataDir + 'voc/'+ dataType + '/' + imgId_ # 组成路径
    cv.imwrite(src, mask) # 保存图片
    txt.write(imgId + '\n')

# initialize COCO api for instance annotations
coco=COCO(annFile) # 初始化coco类
''' # 输出所有分类名
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds()) # 获取类别列表,列表中元素是存储着类名、id和亚类的字典
nms=[cat['name'] for cat in cats] # 遍历类别列表，取出所有类的name
print('COCO categories: \n{}\n'.format(' '.join(nms))) # 打印类名

nms = set([cat['supercategory'] for cat in cats]) # set无序不重复序列 取出所有亚类
print('COCO supercategories: \n{}'.format(' '.join(nms))) # 打印类名
'''
# get all images containing given categories
catIds = coco.getCatIds(catNms=['person']) # 获取指定类别的id
imgIds = coco.getImgIds(catIds=catIds) # 获取包含指定类别id的图像id

# ==========================================
# 遍历满足条件的图像
# ==========================================
txt = open(dataDir + 'voc/'+ dataType + '.txt', "w") # 创建txt记录转换的图片id
print("transforming...")
count = 0
for i in imgIds:
    img = coco.loadImgs(i)[0] # 获取第一个图像的json
    # load and display image
    I = cv.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))  # 从本地加载图像
    # I = cv.cvtColor(I, cv.COLOR_BGR2RGB) # 转换颜色格式
    # use url to load image
    # I = cv.imread(img['coco_url']) # 通过网络从远程加载图

    # ==============================================
    # load and display instance annotations
    # ==============================================
    imgShape = I.shape # 得到图片尺寸
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 得到指定图像对应类型的annotation id
    anns = coco.loadAnns(annIds) # 根据annotation id加载蒙版json

    color = [obj_class, obj_class, obj_class] # 蒙版颜色
    coco2voc(imgShape, anns, color)
    count = count + 1
txt.close()
print(f"transform completely. total:{count}")