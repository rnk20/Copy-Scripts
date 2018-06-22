import tensorflow as tf
import numpy as np
import pandas as pd
import os
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def read_dataset():
    df = pd.read_csv(os.getcwd + '/train.csv')
    X = df.iloc[:,2:]
    Y = df.iloc[:,1]
    return X,Y
X,Y = read_dataset()

print(X.shape)
print(Y.shape)
X, Y =shuffle(X, Y,random_state=45)
#train_x,test_x,train_y,test_y = train_test_split(X, Y,test_size=0.20,random_state=415)


learning_rate=0.3
training_epochs=10
n_features = X.shape[1]
n_samples = X.shape[0]
print("n_features= ",n_features)

n_hidden_1 = 100
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 100

x = tf.placeholder(tf.float64,[None,n_features])
y_ = tf.placeholder(tf.float64,[None])

weigths = {
    'h1': tf.Variable(tf.truncated_normal([n_features, n_hidden_1],dtype=tf.float64)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],dtype=tf.float64)),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],dtype=tf.float64)),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],dtype=tf.float64)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, 1],dtype=tf.float64))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1],dtype=tf.float64)),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2],dtype=tf.float64)),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3],dtype=tf.float64)),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4],dtype=tf.float64)),
    'out': tf.Variable(tf.truncated_normal([1],dtype=tf.float64))
}

def multilayer_perceptron(x,weigths,biases):
    
    layer_1 = tf.add(tf.matmul(x, weigths['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weigths['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weigths['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    layer_4 = tf.add(tf.matmul(layer_3, weigths['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    out_layer = tf.matmul(layer_4, weigths['out']) + biases['out']
    return out_layer

y = multilayer_perceptron(x,weigths,biases)

cost_function = tf.reduce_sum(tf.pow(y-y_, 2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mse_history = []

for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict = {x:X, y_:Y})
    
    mse = sess.run(cost_function, feed_dict = {x:X, y_:Y})
    mse_history.append(mse)
    
    print('epoch= ',epoch,'mse= ',mse)

