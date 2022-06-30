import re
import time

import numpy as np
import open3d as o3d
from pyspark import SparkContext,SparkConf
import numpy as np
import open3d as o3d
import gc
# 把点云找到边角建立坐标系归0
import pandas as pd
from matplotlib import pyplot as plt


def zero(pcd_in):
    np_points = np.asarray(pcd_in.points) - np.asarray(pcd_in.points).min(0)
    zero_pcd = o3d.geometry.PointCloud()
    zero_pcd.points = o3d.utility.Vector3dVector(np_points)
    zero_pcd.colors = pcd_in.colors

    return zero_pcd

# 把numpy转pcd
def np_trans2pcd(np_points, np_colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    return pcd

def np2pcd(np_pcd):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(np_pcd[:, 3:6])
    return pcd


def pcd2np(pcd_in):
    np_points = np.asarray(pcd_in.points)
    np_colors = np.asarray(pcd_in.colors)
    np_pcd = np.hstack((np_points,np_colors))
    return np_pcd

# 计算高度
def cal_height(pcd, avg_x, avg_y, size, h_ratio=0.1):
    height = np.zeros((size, size))
    np_points = np.asarray(pcd.points)
    min = np.min(np_points, 0)
    h_max = np.max(np_points[:, 2])
    np_points[:, 0] = np_points[:, 0] - min[0]
    np_points[:, 1] = np_points[:, 1] - min[1]

    len_x = avg_x / size
    len_y = avg_y / size

    np_points = np_points[np_points[:, 0].argsort()]  # 对x排序
    i = 0
    j = 0
    for x in range(size):
        i = 0
        if np_points.shape[0] == 0:
            break
        while x * len_x <= np_points[i][0] < (x + 1) * len_x:
            i += 1
            if i >= np_points.shape[0]:
                break

        sild = np_points[0:i]
        sild = sild[sild[:, 1].argsort()]

        # sild中个数不为0
        if i != 0:
            np_points = np.delete(np_points, np.arange(i), axis=0)

        for y in range(size):
            j = 0
            if sild.shape[0] == 0:
                break

            while y * len_y <= sild[j][1] < (y + 1) * len_y:
                j += 1
                if j >= sild.shape[0]:
                    break

            if j != 0:
                temp = sild[0:j]
                if np.max(temp[:, 2]) >= h_ratio * h_max:
                    height[x][y] = np.max(temp[:, 2])
                sild = np.delete(sild, np.arange(j), axis=0)
            else:
                height[x][y] = 0

    if np.any(height == 0):
        cnt_array = np.where(height, 0, 1)
        count = cnt_array.sum()
        mean = height.sum() / (size ** 2 - count)
    else:
        mean = height.mean()

    return mean

# 均值下采样 到5w点左右
def uniform_sample(pcd_in, count, min):
    pcd_in_count = int(re.sub(r'\D', "", str(pcd_in)))
    for i in range(2, pcd_in_count // min + 1):
        if i * min <= count < (i + 1) * min:
            pcd_sample = pcd_in.uniform_down_sample(i)
            return pcd_sample

    return pcd_in


# 可视化矩阵
def plot_matrix(height, name):
    # 这里是创建一个数据
    height = np.around(height, 3)
    y = ["{}".format(i) for i in range(0, height.shape[0])]
    x = ["{}".format(i) for i in range(0, height.shape[1])]

    # 这里是创建一个画布
    fig, ax = plt.subplots(figsize=(7, 14), dpi=400)  # 5/27
    im = ax.imshow(height, cmap="YlGn")  # "YlGn"--绿色，Reds红色

    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    # ... and label theym with the respective list entries
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, height[i, j],
                           ha="center", va="center", color="black", size="7")
    ax.set_title("Block " + name.split(".")[0])
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(name)
    plt.show()


#保存为csv
def save_csv(path, numpy_array):
    dataframe = pd.DataFrame(numpy_array)
    dataframe.to_csv(path)


def cal_para(pcd_in, para):
    max_xyz = np.asarray(pcd_in.points).max(0)
    print("区域大小为 : {}".format(max_xyz))
    avg_x = (max_xyz[0] - 0.3) / para[0]
    count_y = int(max_xyz[1] // para[1])  # 向下取整
    print("max_xyz : {}".format(max_xyz))
    print("avg_x : {0},count_y : {1}".format(avg_x, count_y))

    shape = (count_y, para[0])
    height = np.zeros(shape)
    pcd_list = []
    volume = np.zeros(shape)
    surface = np.zeros(shape)
    inclination = np.zeros(shape)
    g_scale = np.zeros(shape)
    canopy_ratio = np.zeros(shape)
    leaf_roll = np.zeros(shape)

    points = np.zeros(shape)
    # 划分y--长
    # for i in range(count_y):
    #     np_points = np.asarray(pcd_in.points)
    #     np_colors = np.asarray(pcd_in.colors)
    #
    #     bool_y = np.logical_and(np_points[:, 1] >= i * 0.8+0.1, np_points[:, 1] < (i + 1) * 0.8-0.1)
    #     y = np_points[bool_y]
    #     colors_y = np_colors[bool_y]
    #     i += 1
    #     # 划分x--宽

    np_points = np.asarray(pcd_in.points)
    np_colors = np.asarray(pcd_in.colors)
    np_pcd = np.concatenate([np_points, np_colors], axis=1)
    np_pcd = np_pcd[np.argsort(np_pcd[:, 1])]
    m, n = 0, 0
    len = np_pcd.shape[0]

    for i in range(count_y):
        for k in range(m,len):
            if np_pcd[k,1] <= (i + 1) * 0.8:
                n=k+1
            else:
                np_pcd_y = np_pcd[m:n]
                m = n
                break


        for j in range(para[0]):
            if j < (para[0] / 2):
                bool_xy = np.logical_and(np_pcd_y[:, 0] >= j * avg_x, np_pcd_y[:, 0] < (j + 1) * avg_x)
            else:
                bool_xy = np.logical_and(np_pcd_y[:, 0] >= j * avg_x + 0.3, np_pcd_y[:, 0] < (j + 1) * avg_x + 0.3)
            # print("j * avg_x : {0}".format(j * avg_x))
            xy = np_pcd_y[bool_xy]

            pcd_xy = np2pcd(xy)
            np_xy = pcd2np(pcd_xy)

            print("i j : {0},{1} ".format(i, j))
            # h = cal_height(pcd_xy, avg_x, 0.8, 5, h_ratio=0.1)  # 计算最高值0.1以上的高度
            # height[i - 1, j] = h
            pcd_list.append(np_xy)


            # 保存第一行
            # if i <=1:
            #     o3d.io.write_point_cloud("pcd/{0}.pcd".format(j),pcd_z)
            # #降采样--点少于一定程度---随机下采样效果差
            # if np_points_z.shape[0] >= 200000:
            #     np_points_sample_z, sample_index = random_sample(np_points_z, 200000)
            #     np_colors_sample_z = np_colors_z[sample_index]
            #     pcd_z = np_trans2pcd(np_points_sample_z, np_colors_sample_z)

            # #均值下采样
            # if np_points_z.shape[0] >= 100000:
            #     pcd_z = o3d.geometry.PointCloud.uniform_down_sample(pcd_z, 4)
            #


            j += 1
    # return height
    return pcd_list,avg_x,shape

# 矩阵左右互换，和实际位置对应
def reverse_matrix(matrix):
    return np.flip(matrix, 1)



if __name__ == '__main__':
    old_time_start = time.time()
    pcd_read = o3d.io.read_point_cloud("2.pcd")
    # 记录时间
    old_time = time.time()
    print("运行时间为" + str(old_time - old_time_start) + "s")
    len_xy = (8, 0.8, 0.3)  # 划分小方格--自定义--y长0.8，x划分8个，中间水沟0.3
    pcd_zero = zero(pcd_read)
    pcd_list, avg_x, shape = cal_para(pcd_zero, len_xy)
    print(shape)
    time1 = time.time()
    print("运行时间为" + str(time1 - old_time) + "s")
    data_list = pcd_list
    # release mem
    del pcd_read
    gc.collect()
    # slice = data_list[100:]

    # 1) 创建SparkContext对象
    conf = SparkConf().setAppName('pcd').setMaster('local[*]')
    conf.set("spark.executor.memory", "64g")
    conf.set("spark.driver.memory", "64g")
    conf.set("spark.executor.cores", "12")
    conf.set('spark.driver.maxResultsSize', '0')

    sc = SparkContext(conf=conf)

    # sc.addPyFile("function.py")

    rdd = sc.parallelize(data_list)
    rdd_map = rdd.map(lambda np_pcd: cal_height(np2pcd(np_pcd), avg_x, 0.8, 5, h_ratio=0.1))

    # rdd = sc.parallelize(num_list)
    # rdd_map = rdd.flatMap(lambda x: (x,x**2))

    # print(rdd_map.collect())

    height = np.asarray(rdd_map.collect()).reshape(shape)
    print(height)
    time2 = time.time()
    print("运行时间为" + str(time2 - old_time) + "s")

    height_rev = reverse_matrix(height)
    # plot_matrix(height_reves , "result/height.png")
    # save_csv(height_rev, "result/height.csv")
    np.savetxt("result/height.csv", height_rev)
    current_time = time.time()
    print("运行时间为" + str(current_time - old_time) + "s")
    # plot_matrix(height, "result/height.png")
    # save_csv(height, "result/height2.csv")

    # o3d.visualization.draw_geometries([pcd_read])

