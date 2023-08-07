# 5import tensorflow as tf
import numpy as np
from PIL import Image

# 加载预训练的图像分类模型
model = tf.keras.applications.MobileNetV2()

# 加载图像
image_path = 'path_to_your_image.jpg'
image = Image.open(image_path)

# 调整图像大小以适应模型的要求
image = image.resize((224, 224))

# 转换图像为模型所需的格式
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.keras.applications.mobilenet.preprocess_input(image[np.newaxis, ...])

# 对图像进行分类
predictions = model.predict(image)
decoded_predictions = tf.keras.applications.mobilenet.decode_predictions(predictions, top=5)[0]

# 打印预测结果
print("预测结果：")
for pred in decoded_predictions:
    class_name, description, probability = pred
    print(f"{class_name}: {probability * 100:.2f}%")
