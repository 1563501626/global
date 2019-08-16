import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义一张新的图
g = tf.Graph()
c = tf.constant(1.0)
with g.as_default():
    d = tf.constant(0.1)
    print(c.graph)
    print(d.graph)

print("*********")
a = tf.constant(5.0)
b = tf.constant(6.0)
sum = tf.add(a, b)
graph = tf.get_default_graph()
print(graph)

with tf.Session() as session:
    print(session.run(sum))
    print(a.graph)
    print(sum.graph)
    print(session.graph)