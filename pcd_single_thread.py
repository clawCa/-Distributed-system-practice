import copy
import gc
import math
import os
import re
import time

# import cv2
import pandas as pd
# from skimage import measure
import numpy
import open3d as o3d
import numpy as np
import random

import itertools
import matplotlib.pyplot as plt


# 把点云找到边角建立坐标系归0
def zero(pcd_in):
    np_points = np.asarray(pcd_in.points) - np.asarray(pcd_in.points).min(0)
    zero_pcd = o3d.geometry.PointCloud()
    zero_pcd.points = o3d.utility.Vector3dVector(np_points)
    zero_pcd.colors = pcd_in.colors

    return zero_pcd


# 读取bin 转为pcd格式
#无用
def read_point_cloud_bin(bin_path):
    data = np.fromfile(bin_path, dtype=np.float32)

    # format:
    N, D = data.shape[0] // 6, 6
    point_cloud_with_normal = np.reshape(data, (N, D))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 0:3])
    point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 3:6])

    return point_cloud


# shape: 划分8,0.8,3
# height:提取区域max高度
# g_scale:计算计算绿色分量 G/R+G+B
# h / 2--定义冠层高度
def cal_para(pcd_in, para):
    max_xyz = np.asarray(pcd_in.points).max(0)
    print("区域大小为 : {}".format(max_xyz))
    avg_x = (max_xyz[0] - 0.3) / para[0]
    count_y = int(max_xyz[1] // para[1])  # 向下取整
    print("max_xyz : {}".format(max_xyz))
    print("avg_x : {0},count_y : {1}".format(avg_x, count_y))

    shape = (count_y, para[0])
    height = np.zeros(shape)
    volume = np.zeros(shape)
    surface = np.zeros(shape)
    inclination = np.zeros(shape)
    g_scale = np.zeros(shape)
    canopy_ratio = np.zeros(shape)
    leaf_roll = np.zeros(shape)

    points = np.zeros(shape)
    # 划分y--长
    for i in range(count_y):
        np_points = np.asarray(pcd_in.points)
        np_colors = np.asarray(pcd_in.colors)

        bool_y = np.logical_and(np_points[:, 1] >= i * 0.8+0.1, np_points[:, 1] < (i + 1) * 0.8-0.1)
        y = np_points[bool_y]
        colors_y = np_colors[bool_y]
        i += 1
        # 划分x--宽
        for j in range(para[0]):
            if j < (para[0] / 2):
                bool_xy = np.logical_and(y[:, 0] >= j * avg_x, y[:, 0] < (j + 1) * avg_x)
            else:
                bool_xy = np.logical_and(y[:, 0] >= j * avg_x + 0.3, y[:, 0] < (j + 1) * avg_x + 0.3)
            # print("j * avg_x : {0}".format(j * avg_x))
            xy = y[bool_xy]
            # print("xy.shape : {0}".format(xy.shape))
            colors_xy = colors_y[bool_xy]
            # 保存小区
            # if i == 6 and j == 7:
            #     temp_pcd = np_trans2pcd(xy, colors_xy)
            #     o3d.io.write_point_cloud("temp/57.pcd", temp_pcd)
            #     # 保存小区
            # if i == 6 and j == 6:
            #     temp_pcd = np_trans2pcd(xy, colors_xy)
            #     o3d.io.write_point_cloud("temp/56.pcd", temp_pcd)
            pcd_xy = np_trans2pcd(xy, colors_xy)

            print("i j : {0},{1} ".format(i, j))
            h = cal_height(pcd_xy, avg_x, 0.8, 5, h_ratio=0.1)  # 计算最高值0.1以上的高度
            height[i - 1, j] = h

            bool_xyz = xy[:, 2] > 0.6 * h  # h/2以上认为是冠层--超参
            np_colors_z = colors_xy[bool_xyz]  # 高度达到h/2以上的点
            np_points_z = xy[bool_xyz]
            pcd_z = np_trans2pcd(np_points_z, np_colors_z)

            # 计算参数
            # g_scale[i - 1, j] = np_colors_z[:, 1].sum() / np_colors_z.sum()
            pcd_z = uniform_sample(pcd_z, np_points_z.shape[0], 50000)  # 降采样到5w到10w之间

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
            #保存小区
            # if i == 1 and j == 0:
            #     o3d.io.write_point_cloud("temp/00z.pcd", pcd_z)
            # 计算体素后的体积--参数pcd,vexel_size
            # volume_ij = cal_volume(pcd_z, voxel_size_in=0.002)
            # volume[i - 1, j] = volume_ij

            inclination[i - 1, j] = cal_mesh(pcd_z, voxel_size=0.002, alpha=0.01)

            # leaf_roll[i - 1, j], canopy_ratio[i - 1, j] = cal_roll_canopy(pcd_z, size=0.001)

            j += 1
    return height, inclination


# # 去除空洞
# def remove_small_points(image, threshold_point):
#     # img = cv2.imread(image, 0)  # 输入的二值图像
#     img_label, num = measure.label(image, neighbors=8, return_num=True)  # 输出二值图像中所有的连通域
#     props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
#
#     resMatrix = np.zeros(img_label.shape)
#     for i in range(1, len(props)):
#         if props[i].area > threshold_point:
#             tmp = (img_label == i + 1).astype(np.uint8)
#             resMatrix += tmp  # 组合所有符合条件的连通域
#     resMatrix *= 255
#     return resMatrix


# 计算卷叶程度：面积/周长
# 计算冠层覆盖率
# def cal_roll_canopy(pcd, size=0.001, min_area=200, max_area=5000):
#     np_2d = sorted(pcd)
#     shape = np.ceil(np_2d.max(0) / size)
#     shape = shape.astype(int)
#
#     bin_image = np.zeros(tuple(shape), dtype="int32")
#     i, j, k = 0, 0, 0
#     for x in range(shape[0]):
#         j = 0
#
#         while x * size <= np_2d[j][0] < (x + 1) * size:
#             j += 1
#             if j >= np_2d.shape[0]:
#                 break
#         sild = np_2d[0:j]
#         sild = sild[sild[:, 1].argsort()]
#
#         if j != 0:
#             np_2d = np.delete(np_2d, np.arange(j), axis=0)
#
#         for y in range(shape[1]):
#             k = 0
#             if sild.shape[0] == 0:
#                 break
#
#             while y * size <= sild[k][1] < (y + 1) * size:
#                 bin_image[x, y] = 1
#                 k += 1
#                 if k >= sild.shape[0]:
#                     break
#
#             if k != 0:
#                 sild = np.delete(sild, np.arange(k), axis=0)
#
#     bin_image = bin_image * 255
#     bin_image.dtype = np.uint8
#     kernel = np.asarray([[0, 1, 0],
#                          [1, 1, 1],
#                          [0, 1, 0]])
#     kernel1 = np.ones((1, 1), dtype="int32")
#     kernel2 = np.ones((2, 2), dtype="int32")
#     kernel3 = np.ones((3, 3), dtype="int32")
#     kernel4 = np.ones((4, 4), dtype="int32")
#     kernel_cov = np.ones((3, 3), np.float32) / 10
#
#     img = cv2.filter2D(bin_image, -1, kernel3)
#     img = cv2.dilate(img, kernel2, iterations=1)
#     img = cv2.erode(img, kernel2, iterations=1)
#
#     img = img / 255
#     # 冠层覆盖率
#     canopy_ratio = img.sum() / img.size
#     img = remove_small_points(img, 500)
#     # img = cv2.erode(img, kernel2, iterations=1)
#     # img = cv2.filter2D(img, -1, kernel3)
#     img = np.asarray(img, dtype=np.uint8)
#     # 获取轮廓contours
#     contours, layer_num = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     list_area = np.asarray([])
#     list_len = np.asarray([])
#     for i in range(len(contours)):
#         area = cv2.contourArea(contours[i])
#         if area <= min_area:
#             continue
#         elif area >= max_area:
#             continue
#         length = cv2.arcLength(contours[i], True)
#         list_area = np.append(list_area, area)
#         list_len = np.append(list_len, length)
#         # print("area:{}".format(area))
#         # print("卷叶：{}".format(area / length))
#     mean_leaf_roll = np.nanmean(list_area / list_len)
#
#     return mean_leaf_roll, canopy_ratio


# xy归0后排序
def sorted(pcd):
    np_points = np.asarray(pcd.points)
    np_xy = np_points[:, 0:2]
    np_2d = np_xy - np_xy.min(0)

    idex = np.lexsort([np_2d[:, 1], np_2d[:, 0]])
    sorted_num_all = np_2d[idex, :]
    return sorted_num_all


# 计算冠层覆盖率，可输出图片,size=0.005 pixel大约160*160
def cal_canopy_ratio(pcd, size):
    np_2d = sorted(pcd)
    shape = np.ceil(np_2d.max(0) / size)
    shape = shape.astype(int)

    bin_image = np.zeros(tuple(shape), dtype="int32")
    i, j, k = 0, 0, 0
    for x in range(shape[0]):
        j = 0

        while x * size <= np_2d[j][0] < (x + 1) * size:
            j += 1
            if j >= np_2d.shape[0]:
                break
        sild = np_2d[0:j]
        sild = sild[sild[:, 1].argsort()]

        if j != 0:
            np_2d = np.delete(np_2d, np.arange(j), axis=0)

        for y in range(shape[1]):
            k = 0
            if sild.shape[0] == 0:
                break

            while y * size <= sild[k][1] < (y + 1) * size:
                bin_image[x, y] = 1
                k += 1
                if k >= sild.shape[0]:
                    break

            if k != 0:
                sild = np.delete(sild, np.arange(k), axis=0)

    canopy_ratio = bin_image.sum() / bin_image.size

    # #保存图片
    # bin_image = bin_image*255
    # cv2.imwrite("pcd/img.jpg",bin_image)/
    return canopy_ratio


# 均值下采样 到5w点左右
def uniform_sample(pcd_in, count, min):
    pcd_in_count = int(re.sub(r'\D', "", str(pcd_in)))
    for i in range(2, pcd_in_count // min + 1):
        if i * min <= count < (i + 1) * min:
            pcd_sample = pcd_in.uniform_down_sample(i)
            return pcd_sample

    return pcd_in


# 随机下采样到 k 个点，--可返回结果 和 index
def random_sample(np_points, k):
    np_choice = np.arange(np_points.shape[0])
    index = np.random.choice(np_choice, size=5)
    choice_points = np_points[index]

    return choice_points, index


# 给一个向量计算向量与Z轴夹角，0 1 2--对应x y z坐标
def cal_theta(array):
    if array[2] < 0:
        z = -array[2]
    else:
        z = array[2]
    return math.acos(z / math.sqrt((array ** 2).sum())) * 180 / math.pi
    # return math.asin(math.sqrt(array[0]*array[0] + array[1]*array[1]) / math.sqrt((array ** 2).sum())) * 180 / math.pi
    # return math.acos( array[2] / math.sqrt((array ** 2).sum())) * 180 / math.pi


# 计算mesh化后的 表面积和叶倾角  voxel_size 0.005  alpha = 0.008--
#             0.002         0.009
def cal_mesh(pcd_in, voxel_size, alpha):
    pcd_down = pcd_in.voxel_down_sample(voxel_size)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_down, alpha)
    # surface_area = mesh.get_surface_area()

    o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
    np_mesh_vertices = np.asarray(mesh.vertices)
    np_triangles = np.asarray(mesh.triangles)
    np_normals = np.asarray(mesh.triangle_normals)

    S = np.zeros(np_normals.shape[0], dtype="double")
    leaf_inclination = np.zeros(np_normals.shape[0], dtype="double")
    theta = np.zeros(np_normals.shape[0])

    for i in range(np_triangles.shape[0]):
        triangle = np_mesh_vertices[np_triangles[i]]
        AB = triangle[0] - triangle[2]
        AC = triangle[1] - triangle[2]
        s = np.array([AB[1] * AC[2] - AB[2] * AC[1], AB[2] * AC[0] - AB[0] * AC[2], AB[0] * AC[1] - AB[1] * AC[0]])
        S[i] = 0.5 * math.sqrt((s ** 2).sum())

        theta[i] = cal_theta(np_normals[i])
        if theta[i] < 0:
            theta[i] = - theta[i]

        leaf_inclination[i] = theta[i] * S[i]

    dip_angle = leaf_inclination.sum() / S.sum()
    # print("叶倾角theta {0}".format(leaf_inclination.sum() / S.sum()))

    # return surface_area, dip_angle
    return dip_angle

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


# 把numpy转pcd
def np_trans2pcd(np_points, np_colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    return pcd


# 计算体素化后的体积
def cal_volume(pcd, voxel_size_in):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=voxel_size_in)
    voxel_grid_count = int(re.sub(r'\D', "", str(voxel_grid)))
    volume = voxel_grid_count * voxel_size_in ** 2
    return volume


# 计算每一行 所有x区域中的平均高度
def avg_height_x(height):
    return height.sum(1) / height.shape[1]


# 计算每一列y中的平均高度
def avg_height_y(height):
    return height.sum(0) / height.shape[0]


# 计算总体平均高度
def avg_height(height):
    return height.mean()


# 计算绿色分量 G/R+G+B
def cal_g_scale(pcd_in, height):
    np_points = np.asarray(pcd_in.points)


# 矩阵左右互换，和实际位置对应
def reverse_matrix(matrix):
    return numpy.flip(matrix, 1)


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
def save_csv(path,numpy_array):
    dataframe = pd.DataFrame(numpy_array)
    dataframe.to_csv(path, header=False, index=False, sep=',')

if __name__ == '__main__':

    pcd_read = o3d.io.read_point_cloud("819-1 - start-cut-pre.pcd")
    # 记录时间
    old_time = time.time()
    # pcd_read = o3d.io.read_point_cloud("data/819-1 - start-cut-pre.pcd")
    len_xy = (8, 0.8, 0.3)  # 划分小方格--自定义--y长0.8，x划分8个，中间水沟0.3

    pcd_zero = zero(pcd_read)
    del pcd_read
    gc.collect()

    height,inclination = cal_para(pcd_zero, len_xy)
    # print("宽x: {0}, 长y: {1} ,总体平均: {2}".format(avg_height_x(height), avg_height_y(height), avg_height(height)))

    height_rev = reverse_matrix(height)
    inclination_rev = reverse_matrix(inclination)
    # plot_matrix(height_rev, "result2/height.png")
    np.savetxt("result2/height.csv", height_rev)
    np.savetxt("result2/inclination.csv", inclination_rev)
    # save_csv(height,"result/height.csv")

    current_time = time.time()
    print("运行时间为" + str(current_time - old_time) + "s")

    # label = ["{}".format(i) for i in range(0, height.shape[0])]
    # plot_confusion_matrix(height, label)
    # print("x y z : ", np.asarray(pcd_zero.points).max(0))
    # print("min : {0}".format(np.max(np_points, 0)))
