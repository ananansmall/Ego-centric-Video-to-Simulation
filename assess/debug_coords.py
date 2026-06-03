#!/usr/bin/env python3
"""
调试脚本：查看坐标系转换问题
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh
import pyrender
import cv2

def main():
    # 测试坐标系转换
    print("=== 测试坐标系转换 ===")
    
    # main.py 中的 z-up -> y-up 矩阵
    zup_to_yup = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, -1, 0, 0],[0, 0, 0, 1]])
    print("main.py z-up -> y-up:")
    print(zup_to_yup)
    
    # 评估代码中的 y-up -> z-up 矩阵
    yup_to_zup = np.array([[1, 0, 0, 0],[0, 0, -1, 0],[0, 1, 0, 0],[0, 0, 0, 1]])
    print("\n评估代码 y-up -> z-up:")
    print(yup_to_zup)
    
    # 测试是否互逆
    product = zup_to_yup @ yup_to_zup
    print("\n两者相乘（应该是单位矩阵）:")
    print(product)
    
    # 测试一个点
    p = np.array([0, 1, 0, 1])  # z-up 中的 Y 轴
    p_yup = zup_to_yup @ p
    print(f"\n测试点 z-up: {p}")
    print(f"转换到 y-up: {p_yup}")
    print(f"转换回 z-up: {yup_to_zup @ p_yup}")
    
    # 测试点云
    print("\n=== 测试点云 ===")
    # 创建一个简单的立方体（z-up）
    cube = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    print(f"原始立方体中心: {cube.centroid}")
    
    # 应用 z-up -> y-up
    cube_yup = cube.copy()
    cube_yup.apply_transform(zup_to_yup)
    print(f"z-up -> y-up 后中心: {cube_yup.centroid}")
    
    # 应用 y-up -> z-up
    cube_zup_back = cube_yup.copy()
    cube_zup_back.apply_transform(yup_to_zup)
    print(f"y-up -> z-up 后中心: {cube_zup_back.centroid}")
    
    print("\n✓ 坐标系转换验证完成！")

if __name__ == "__main__":
    main()

