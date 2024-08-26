# 图像拼接sift方法，两张图片
import cv2
import numpy as np
import time


def stitch_images(left_image_path, right_image_path):
    start_time = time.time()

    left = cv2.imread(left_image_path)
    right = cv2.imread(right_image_path)

    gray1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    print("正在提取特征点...")
    # 提取特征点和描述符
    kpsA, dpA = sift.detectAndCompute(gray1, None)
    kpsB, dpB = sift.detectAndCompute(gray2, None)

    # 创建 FLANN 特征匹配对象
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(dpA, dpB, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        kps1 = np.float32([kpsA[m.queryIdx].pt for m in good_matches])
        kps2 = np.float32([kpsB[m.trainIdx].pt for m in good_matches])

        M, mask = cv2.findHomography(kps2, kps1, cv2.RANSAC, 5.0)

        result_height = max(left.shape[0], right.shape[0])
        result_width = left.shape[1] + right.shape[1]
        result = cv2.warpPerspective(right, M, (result_width, result_height))

        # 将左图的像素放置在变换后右图的位置
        result[0:left.shape[0], 0:left.shape[1]] = left

        # 代码执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        print("代码执行时间：", execution_time, "秒")

        # 裁剪图像，去除空白区域
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_result = result[y:y + h, x:x + w]

        return cropped_result
    else:
        print("匹配点不足，无法拼接图像。")
        return None


if __name__ == '__main__':
    left_path = "test/test1.jpg"
    right_path = "test/test2.jpg"
    result = stitch_images(left_path, right_path)

    if result is not None:
        # 保存
        cv2.imwrite("test/r2.jpg", result)

        left = cv2.imread(left_path)
        right = cv2.imread(right_path)

        # 调整宽度使图像可以水平堆叠
        max_height = max(left.shape[0], right.shape[0], result.shape[0])
        left_resized = cv2.resize(left, (int(left.shape[1] * max_height / left.shape[0]), max_height))
        right_resized = cv2.resize(right, (int(right.shape[1] * max_height / right.shape[0]), max_height))
        result_resized = cv2.resize(result, (int(result.shape[1] * max_height / result.shape[0]), max_height))

        # 可调整大小的窗口
        cv2.namedWindow("Images with Keypoints", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Images with Keypoints", 2000, 400)
        cv2.imshow("Images with Keypoints", np.hstack((left_resized, right_resized, result_resized)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()