# tf.compat.v1.disable_eager_execution()
# img = tf.compat.v1.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
# const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
# val = img + const
# out = tf.identity(val, name="out")

# # Convert to TF Lite format
# with tf.compat.v1.Session() as sess:
#   converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [img], [out])
#   tflite_model = converter.convert()
# with open('model.tflite', 'w') as f:
#     f.write(str(tflite_model))
 
from jnius import autoclass
c = (2 for i in range(2))
print(c)
a, b = c
