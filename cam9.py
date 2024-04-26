import cv2 as cv

# 创建摄像头对象
camera = cv.VideoCapture(0)  # 参数为摄像头索引，0表示第一个摄像头

# 检查摄像头是否成功打开
if not camera.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取并显示摄像头的图像
while True:
    ret, frame = camera.read()  # 读取图像帧
    if not ret:
        print("无法获取摄像头的图像")
        break

    cv.imshow("Camera", frame)  # 显示图像

    # 按下 'q' 键退出循环
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
camera.release()
cv.destroyAllWindows()