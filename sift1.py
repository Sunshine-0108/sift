# 连续拼接一个文件夹中所有图片
import cv2
import numpy as np
import os
import time


def stitch_images(left, right):
    gray1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kpsA, dpA = sift.detectAndCompute(gray1, None)
    kpsB, dpB = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用 KNN 进行特征匹配
    matches = flann.knnMatch(dpA, dpB, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        kps1 = np.float32([kpsA[m.queryIdx].pt for m in good_matches])
        kps2 = np.float32([kpsB[m.trainIdx].pt for m in good_matches])

        # 单应矩阵
        M, mask = cv2.findHomography(kps2, kps1, cv2.RANSAC, 5.0)

        result_height = max(left.shape[0], right.shape[0])
        result_width = left.shape[1] + right.shape[1]
        result = cv2.warpPerspective(right, M, (result_width, result_height))

        # 将左图的像素放置在变换后右图的位置
        result[0:left.shape[0], 0:left.shape[1]] = left

        # 裁剪图像，去除空白区域
        # gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # x, y, w, h = cv2.boundingRect(contours[0])
        # cropped_result = result[y:y + h, x:x + w]

        return result
    else:
        print("匹配点不足，无法拼接图像。")
        return None


def stitch_folder_images(folder_path):
    # 获取文件夹中所有图片的文件名，按数字顺序排序
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))

    # 读取第一张图片作为基础图片
    base_image = cv2.imread(os.path.join(folder_path, image_files[0]))

    # 从第二张图片开始，依次与基础图片拼接
    for i in range(1, len(image_files)):
        right_image_path = os.path.join(folder_path, image_files[i])
        right_image = cv2.imread(right_image_path)

        print(f"正在拼接: {image_files[i - 1]} 和 {image_files[i]}")

        # 拼接两张图片
        stitched_image = stitch_images(base_image, right_image)

        if stitched_image is not None:
            # 更新基础图片为拼接结果
            base_image = stitched_image
            # 保存拼接后的结果
            cv2.imwrite(f"result_{i}.jpg", base_image)
        else:
            print(f"图像 {image_files[i]} 拼接失败，跳过该图像。")

    # 最后返回最终拼接完成的图像
    return base_image


if __name__ == '__main__':
    folder_path = "try9"
    final_result = stitch_folder_images(folder_path)

    if final_result is not None:
        # 保存最终结果
        cv2.imwrite("try9/final_stitched_image.jpg", final_result)

        # 最终拼接结果
        cv2.namedWindow("Final Stitched Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Final Stitched Image", 2000, 400)
        cv2.imshow("Final Stitched Image", final_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
