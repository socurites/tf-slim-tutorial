# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
native TF에서 스코프를 사용하는 방법
# 1. name_scope
# 2. variable_scope
#
# 두 스코프 방식은 모두 변수앞에 prefix를 붙이는 역할은 동일하나 약간의 차이가 있음
# Ref: https://stackoverflow.com/questions/34215746/difference-between-variable-scope-and-name-scope-in-tensorflow
"""

"""
name_scope()와 variable_scope()의 차이
# 1. tf.variable_scope()는 스코프 내에 있는 모든 변수에 대해 preifx를 추가
#    - tf.Variable(), tf.get_variable() 어느 경우로 변수를 생성해도 동일
# 2. tf.name_scope()는 tf.Variable()로 생성한 변수에 대해서만 preifx를 추가
#    - tf.get_variable()로 생성한 변수는 스코프에 포함되지 않음(즉 prefix가 추가되지 않음)
"""
import tensorflow as tf
import matplotlib.pyplot as plt

"""
name_scope()와 variable_scope()의 차이 비교
"""
def scoping(fn, scope1, scope2, vals):
    with fn(scope1):
        a = tf.Variable(vals[0], name='a')
        b = tf.get_variable('b', initializer=vals[1])
        c = tf.constant(vals[2], name='c')
        with fn(scope2):
            d = tf.add(a * b, c, name='res')

        print '\n  '.join([scope1, a.name, b.name, c.name, d.name]), '\n'
    return d

d1 = scoping(tf.variable_scope, 'scope_vars', 'res', [1, 2, 3])
d2 = scoping(tf.name_scope,     'scope_name', 'res', [1, 2, 3])

with tf.Session() as sess:
    writer = tf.summary.FileWriter('/tmp/tf-slim-tutorial', sess.graph)
    sess.run(tf.global_variables_initializer())
    print sess.run([d1, d2])
    writer.close()

# 텐서보드 실행
# $ tensorboard --logdir=/tmp/tf-slim-tutorial

# 텐서보드 접속
# http://localhost:6006

"""
tf.get_variable
# 이미 정의된 변수를 가져오거나, 없으면 새로 생성한다
# 위의 예제에서는 변수 'b'가 정의가 되어 있지 않으므로 새로 생성하다.
# 변수 'b'가 정의되어 있는 경우 name_scope()와 variable_scope()의 차이
# Ref: https://www.tensorflow.org/api_docs/python/tf/get_variable
"""

b = tf.Variable(initial_value=1, name='b')

def scoping(fn, scope1, scope2, vals):
    with fn(scope1):
        a = tf.Variable(vals[0], name='a')
        b = tf.get_variable('b', initializer=vals[1])
        c = tf.constant(vals[2], name='c')
        with fn(scope2):
            d = tf.add(a * b, c, name='res')

        print '\n  '.join([scope1, a.name, b.name, c.name, d.name]), '\n'
    return d, b


d1, b1 = scoping(tf.variable_scope, 'scope_vars2', 'res', [1, 2, 3])
# d2 = scoping(tf.name_scope, 'scope_name2', 'res', [1, 2, 3])
# ValueError: Variable b already exists, disallowed
