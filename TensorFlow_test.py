# 测试代码
# https://zhuanlan.zhihu.com/p/37086409
import tensorflow as tf
import cv2

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

img = cv2.imread("img/demo2.jpg")
# img = cv2.imread("D:/ComputerVision/FaceMaskDetection/img/demo2.jpg")
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)