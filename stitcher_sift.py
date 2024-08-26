# 先利用stitcher函数对文件夹中图片拼接为全景图，再通过sift算子算出对应变换矩阵，将所有图片拼接到一起，
# 目的得到单应性矩阵
import cv2
import numpy as np
import os
import re
from typing import List


def extract_number_from_filename(filename: str) -> int:
    # 使用正则表达式提取文件名中的数字
    # 保证按文件名中数字递增排序
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def sort_files_by_number(file_paths: List[str]) -> List[str]:
    # 根据文件名中的数字对文件路径进行排序
    return sorted(file_paths, key=lambda p: extract_number_from_filename(os.path.basename(p)))


def find_max_interior_rectangle(image):
    print("正在找出最大内接矩形...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.boundingRect(largest_contour)
        x, y, w, h = rect
        cropped_image = image[y:y + h, x:x + w]
        print(f"最大内接矩形区域: (x={x}, y={y}, w={w}, h={h})")
        return cropped_image
    else:
        raise ValueError("未找到有效的内接矩形区域")


def stitch_images(image_paths):
    print("正在拼接图像...")
    images = [cv2.imread(p) for p in image_paths]
    stitcher = cv2.Stitcher.create() if cv2.__version__.startswith('4.') else cv2.Stitcher.create(False)
    status, panorama = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("图像拼接成功")
        return panorama
    else:
        raise RuntimeError("Stitching failed with status code: {}".format(status))


def compute_transform_matrices(image_paths, panorama):
    print("正在计算变换矩阵...")
    transforms = []
    panorama_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(image_gray, None)
        kp2, des2 = sift.detectAndCompute(panorama_gray, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            transforms.append(M)
            print(f"计算图像 {image_path} 的变换矩阵成功")
        else:
            raise ValueError(f"Not enough matches found for image: {image_path}")

    return transforms


def transform_and_stitch_images(image_paths, transforms, panorama_shape):
    print("正在将所有图像拼接到一张图中...")
    panorama_h, panorama_w = panorama_shape[:2]

    # 创建一个全零图像来存储拼接结果
    stitched_image = np.zeros((panorama_h, panorama_w, 3), dtype=np.float32)
    count_map = np.zeros((panorama_h, panorama_w), dtype=np.float32)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        M = transforms[i]
        transformed_image = cv2.warpPerspective(image, M, (panorama_w, panorama_h))

        # 记录当前图像的有效区域
        mask = np.any(transformed_image != 0, axis=-1)
        mask_float = mask.astype(np.float32)

        # 如果不是第一张图像，删除与已拼接图像重叠的部分
        if i > 0:
            overlap_mask = (count_map > 0).astype(np.float32)
            mask_float = mask_float * (1 - overlap_mask)

        # 更新拼接图像和计数图像
        stitched_image += transformed_image.astype(np.float32) * mask_float[:, :, np.newaxis]
        count_map += mask_float

    count_map[count_map == 0] = 1
    stitched_image /= count_map[:, :, np.newaxis]

    return np.clip(stitched_image, 0, 255).astype(np.uint8)


def save_transformed_images(image_paths, panorama, transforms, output_folder):
    print("正在保存变换后的图像...")
    panorama_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        M = transforms[i]
        transformed_image = cv2.warpPerspective(image, M, (panorama.shape[1], panorama.shape[0]))
        output_path = os.path.join(output_folder, f"transformed_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, transformed_image)
        print(f"保存图像 {output_path} 成功")


def main(rgb_folder, output_images_folder):
    rgb_files = [os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 按文件名中的数字递增顺序排序
    rgb_files = sort_files_by_number(rgb_files)

    # 拼接图像生成全景图
    panorama = stitch_images(rgb_files)

    # 找出最大内接矩形区域并裁剪图像
    cropped_panorama = find_max_interior_rectangle(panorama)
    cv2.imwrite(os.path.join(output_images_folder, "stitcher_cropped.png"), cropped_panorama)
    print("裁剪后的全景图保存成功")

    # 计算每张图像到裁剪后的全景图的变换矩阵
    transforms = compute_transform_matrices(rgb_files, cropped_panorama)

    # 将所有图像按其对应变换矩阵拼接到一张图中
    stitched_image = transform_and_stitch_images(rgb_files, transforms, cropped_panorama.shape)
    cv2.imwrite(os.path.join(output_images_folder, "sift_cropped.png"), stitched_image)
    print("裁剪后的拼接图像保存成功")

    # 保存每张RGB图像的变换图
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    save_transformed_images(rgb_files, cropped_panorama, transforms, output_images_folder)


if __name__ == "__main__":
    # rgb图所在文件夹
    rgb_folder = 'try9_rgb'
    # 输出的变换图像文件夹
    output_images_folder = 'try9_rgb'

    main(rgb_folder, output_images_folder)
