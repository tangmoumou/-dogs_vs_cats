# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:45:06 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import os

#%%
train_dir=r'E:\AI\python\4MLmodel\tensorflow\CNNproject\2dogs_vs_cats\dataset\train'

cats_path=[]
label_cats=[]
dogs_path=[]
label_dogs=[]
for file in os.listdir(train_dir):
    '''分类依据为文件名的第一个单词'''
    name=file.split('.')
    if name[0] == 'cat':
        cats_path.append(train_dir+r'\\'+file)
        label_cats.append(0)
    else:
        dogs_path.append(train_dir+r'\\'+file)
        label_dogs.append(1)
print('There are %d cats\nThere are %d dogs'%(len(cats_path),len(dogs_path)))
    
image_path_list=np.hstack((cats_path,dogs_path))
label_list=np.hstack((label_cats,label_dogs))

temp=np.array([image_path_list,label_list])
temp=temp.transpose()
np.random.shuffle(temp)
    
image_path_list=list(temp[:,0])
label_list=list(temp[:,1])
label_list=[int(i) for i in label_list]

#%%
image_path=image_path_list
label=label_list
IMAGE_W=208
IMAGE_H=208
BATCH_SIZE=20
CAPACITY=256

image_path=tf.cast(image_path,tf.string)
label=tf.cast(label,tf.int32)
    
#make an input queue
input_queue=tf.train.slice_input_producer([image_path,label])
    
label=input_queue[1]
image_contents=tf.read_file(input_queue[0])
image=tf.image.decode_jpeg(image_contents,channels=3)
    
################################################
#data argumentation should go to here
################################################
    
image=tf.image.resize_image_with_crop_or_pad(image,IMAGE_W,IMAGE_H)
#if you want to test the generated batches of images, you might want to comment the following line
#如果想看到正常的图片，请注释掉标准化，和image_batch=tf.cast(image_batch,tf.float32)
#训练时不要注释掉
image=tf.image.per_image_standardization(image)
    
image_batch,label_batch=tf.train.batch([image,label],
                                        batch_size=BATCH_SIZE,
                                        num_threads=64,
                                        capacity=CAPACITY)
    
label_batch=tf.reshape(label_batch,[BATCH_SIZE])
image_batch=tf.cast(image_batch,tf.float32)

#%%
#import matplotlib.pyplot as plt
#with tf.Session() as sess:
#    i=0
#    coord=tf.train.Coordinator()
#    threads=tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img,label=sess.run([image_batch,label_batch])
#            
#            #just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label:%d'%label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)

#%%

#转换样本的形式以便于进行卷积
x_input=image_batch#转换形式是n个纵深的图片，图片是尺寸是28*28*1
##添加dropout
keep_prob=0.5


'''对输入的图像进行第一次卷积'''
#定义卷积函数的参数
#定义权重weight
w_conv1=tf.Variable(tf.truncated_normal([5,5,3,32],stddev=0.1))#权重是可训练参数，使用Variable方法定义，
###并利用截断正态分布赋初值，采用5*5大小的窗口，将输入厚度为1的图片映射成输出为32层的feature map
#定义偏差bias
tf.summary.histogram('w1',w_conv1)
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))#由于tf矩阵运算中的加法具有broadcaster属性，只需要声明输出行数
#进行卷积运算
tf.summary.histogram('b1',b_conv1)
conv1=tf.nn.relu(tf.nn.conv2d(x_input,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)#2d卷积方法，输入x_input，w_conv1，
#######步长2*2，填充至一样大小，此时shape为[-1,208,208,32]
tf.summary.histogram('conv1',conv1)

'''对feature map进行第一次池化'''
pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#最大池化，ksize经验赋值，shape([-1,104,104,32])

'''进行第二次卷积'''
w_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
conv2=tf.nn.relu(tf.nn.conv2d(pool1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)#shape=[-1,104,104,64]

'''进行第二次池化'''
pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#shape=[-1,52,52,64]

'''连接全连接层1'''
#定义全连接层的参数
#定义权重weight,目的是将每个图片的3维tensor特征变为1维数组特征列表，故先对池化后的feature map进行reshape
pool2_f=tf.reshape(pool2,[-1,52*52*64])
w_fullc1=tf.Variable(tf.truncated_normal([52*52*64,1024],stddev=0.1))
b_fullc1=tf.Variable(tf.constant(0.1,shape=[1024]))
fullc1=tf.nn.relu(tf.matmul(pool2_f,w_fullc1)+b_fullc1)#矩阵运算并激活,shape=[-1,7*7*1024]
#使用dropout
fullc1_dropout=tf.nn.dropout(fullc1,keep_prob)

'''连接全连接层2分类'''
w_fullc2=tf.Variable(tf.truncated_normal([1024,2],stddev=0.1))
b_fullc2=tf.Variable(tf.constant(0.1,shape=[2]))
softmax_linear=tf.matmul(fullc1,w_fullc2)+b_fullc2#矩阵运算并激活,shape=[-1,10]

##'''输出预测结果'''
###打开会话
##sess=tf.Session()
###全局变量初始化
##init=tf.global_variables_initializer()
##sess.run(init)
###提共样本
##xs=mnist.index(1,3)
###进行预测
##print (sess.run(prediction))

'''定义损失函数'''
with tf.variable_scope('loss') as scope:
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear, 
                                                                   labels=label_batch,
                                                                   name='xentropy_per_example')#预测值是softmax模式，采用交叉熵作为损失函数
    loss = tf.reduce_mean(cross_entropy,name='loss')
    tf.summary.scalar(scope.name+'/loss', loss)

'''定义训练方法'''
train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)#采用Adam优化算法

'''训练精度'''
correct = tf.nn.in_top_k(softmax_linear, label_batch, 1)
correct = tf.cast(correct, tf.float16)
accuracy = tf.reduce_mean(correct)
'''保存训练'''
saver=tf.train.Saver()

'''打开会话'''
sess=tf.Session()

'''合并summary添加graph'''
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/',sess.graph)

'''全局变量初始化'''
init=tf.global_variables_initializer()
sess.run(init)

'''保存到指定路径的指定文件'''
save_path=saver.save(sess,'my_net/3.ckpt')
print('Save to path:',save_path)

summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs', sess.graph)

'''开始训练'''
#定义训练次数
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

MAX_STEP = 10000   
try: 
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_step, loss, accuracy])
               
        if step % 50 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
            
        if step % 2000 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join('my_net', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
                
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
        
coord.join(threads)
sess.close()


