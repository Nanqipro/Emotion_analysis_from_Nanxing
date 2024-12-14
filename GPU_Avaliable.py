import tensorflow as tf
from tensorflow.python.framework import ops

# 检查 GPU 是否可用
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print("GPU is available.")
else:
    print("No GPU available.")

# 检查 TensorFlow 是否支持 cuDNN
print("TensorFlow version: ", tf.__version__)
print("CUDA version: ", tf.sysconfig.get_cuda_version())
print("cuDNN version: ", tf.sysconfig.get_cudnn_version())
