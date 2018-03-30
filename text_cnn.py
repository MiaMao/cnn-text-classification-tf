import tensorflow as tf
import numpy as np

#一个卷积核对应于一个句子，convolution之后得到一个vector，max——pooling之后得到一个scalar

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            #查找input_x中的所有ids，获取他们的word_vector，batch中的每个sentence的每个word都要查找，
            #所以得到的embedded——chars的shape应该是[None,sequence_length,embedding_size]
            
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #加上一个in_channels = 1
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                #filter_shape = [filter_height,filter_width, in_channels, out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                #input[batc, in_heigth,in_width, in_channels]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #conv2d and relu, shape = [batch, sequence_length-filter_size+1,1, num_filters] 
                # Maxpooling over the outputs
                #max-pooling(value, ksize, strides, padding, data_format = 'NHWC', name= None)
                pooled = tf.nn.max_pool(
                    h,
                    #多大范围内进行max-pooling
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                   #pooled shape=[batch, 1, 1, num_filters]
               #pooled存储的是当前filter_size下每个sentence最重要的num_filters个feature,结果append到pooled——output中
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        #将不同filter的结果concatenate，得到[batch, all_pooled_result],其中，后者的大小应该是num_filters*len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        #dropout仅对hiddenlayer的输出层进行drop，使得有些结点的结果不输给softmax层。
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            #score shape [batch, 1]
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #选取每行的max值，dimension=1
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #reduce_mean()本身输入的就是一个float类型的vector(0.0或者1.0).直接对这样的vector计算mean就是accuracy。
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
#
