import argparse
from pathlib import Path

import cv2
import numpy as np
import math
import os

from obj_loader import *


def read_image(image_file, scale=0.25):
    image = cv2.imread(image_file)
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width*scale), int(height*scale)))
    # 转换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

#渲染模型
def render(image, obj, projection, refer_size, model_scale=1.0,frame_color='#0000FF', fill_color='#555555'):
    scale_matrix = np.eye(3) * model_scale
    width, height = refer_size


    vertices = np.array(obj.vertices)
    vertices = np.dot(vertices, scale_matrix)
    vertices = vertices + np.array([width / 2, height / 2, 0])
    trans_vertices = cv2.perspectiveTransform(vertices.reshape(-1, 1, 3), projection)
    img_points = np.int32(trans_vertices)


    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([img_points[vertex - 1] for vertex in face_vertices])


        cv2.polylines(image, [points], True, (255, 0, 0), 2, -1)

    return image


def get_projection_matrix(camera_matrix, homography):
    """
    从相机内参和单应性推导投影矩阵
    """
    # 内参逆向映射，单应性跨图映射
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_matrix), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # 归一化向量
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # 计算正交基
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    # 计算3D映射矩阵
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_matrix, projection)


def main(image_dir, image_scale, model_path, model_scale, frame_color, fill_color, min_matches,  draw_matches):
    # 读取相机内参

    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # 创建ORB检测器
    orb = cv2.ORB_create()
    # 蛮力 (Brute-Force, BF) 匹配器：该匹配器利用为第一组中检测到的特征计算的描述符与第二组中的所有描述符进行匹配。最后，它返回距离最近的匹配项。
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 加载3D模型
    obj = OBJ(model_path, swapyz=True)

    # 加载参考图片并提取关键点
    refer_image_path = os.path.join(image_dir, 'refer2.jpg')
    refer_image, refer_gray = read_image(refer_image_path, image_scale)
    refer_height, refer_width = refer_gray.shape[:2]
    kp_ref, des_ref = orb.detectAndCompute(refer_gray, None)

    # 读取图片列表
    image_files = [str(x) for x in Path(args.image_dir).glob('*.jpg') if 'refer' not in str(x)]

    for image_file in image_files:
        # 读取图片
        image, gray = read_image(image_file, image_scale)

        # 执行通过 orb.detectAndCompute(image, None) 函数同时检测关键点并计算检测到的关键点的描述符
        kp_tar, des_tar = orb.detectAndCompute(gray, None)
        # 返回值为两个图像中获得了最佳匹配，两个参数是计算得到的描述符
        matches = bf.match(des_ref, des_tar)

        # 根据距离进行排序
        matches = sorted(matches, key=lambda x: x.distance)

        # 关键点匹配数量必须满足数量
        if len(matches) > min_matches:
            # 匹配的关键点在源图像中的位置
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            # 是匹配的关键点在查询图像中的位置
            dst_pts = np.float32([kp_tar[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # RANSAC方法计算单应性矩阵，使用cv2.findHomography()函数在两幅图像中找到匹配关键点位置之间的透视变换
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 如果匹配成功则渲染模型
            if homography is not None:
                projection = get_projection_matrix(camera_matrix, homography)
                image = render(image, obj, projection, (refer_width, refer_height), model_scale, frame_color, fill_color)

            # 使用cv2.drawMatches()函数绘制匹配特征对，水平拼接两个图像，并绘制从第一个图像到第二个图像的线条以显示匹配特征对
            if draw_matches:
                image = cv2.drawMatches(refer_image, kp_ref, image, kp_tar, matches[:min_matches], 0, flags=2)

            # 显示结果
            cv2.imshow('RESULT', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        else:
            print(f"No enough matches found - {len(matches)}/{min_matches}")

    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='../data/card')
    parser.add_argument('--image-scale', type=float, default=0.2, help='image scaling factor')
    parser.add_argument('--model-path', type=str, default='../models/wolf.obj')
    parser.add_argument('--model-scale', type=float, default=0.5, help='default model scale')
    parser.add_argument('--frame-color', type=str, default='#0000FF', help='default model color')
    parser.add_argument('--fill-color', type=str, default='#555555', help='default model color')
    parser.add_argument('--min-matches', type=int, default=100, help='number of minimum matches')
    parser.add_argument('--draw-matches', action="store_true", default=True)
    args = parser.parse_args()

    main(args.image_dir,
        args.image_scale,
        args.model_path,
        args.model_scale,
        args.frame_color,
        args.fill_color,
        args.min_matches,
        args.draw_matches)
