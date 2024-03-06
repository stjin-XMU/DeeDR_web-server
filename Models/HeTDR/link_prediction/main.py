import math
import os
import sys
import time

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tqdm
from numpy import random

from utils_hw import *


def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32), np.array(neigh).astype(np.int32)) 

def train_model(network_data, feature_dic, log_name):
    all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema)
    vocab, index2word = generate_vocab(all_walks)
    train_pairs = generate_pairs(all_walks, vocab, args.window_size)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions # Dimension of the embedding vector.
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples # Number of negative examples to sample.
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples 

    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples-len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))

    graph = tf.Graph()

    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if feature_dic is not None:
            node_features = tf.Variable(features, name='node_features', trainable=False)
            feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0))
            linear = tf.layers.Dense(units=embedding_size, activation=tf.nn.tanh, use_bias=True)

            embed_trans = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            u_embed_trans = tf.Variable(tf.truncated_normal([edge_type_count, feature_dim, embedding_u_size], stddev=1.0 / math.sqrt(embedding_size)))

        # Parameters to learn
        node_embeddings = tf.Variable(tf.random.uniform([num_nodes, embedding_size], -1.0, 1.0))
        node_type_embeddings = tf.Variable(tf.random.uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0))
        trans_weights = tf.Variable(tf.random.truncated_normal([edge_type_count, embedding_u_size, embedding_size // att_head], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s1 = tf.Variable(tf.random.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s2 = tf.Variable(tf.random.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.random.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        node_neigh = tf.placeholder(tf.int32, shape=[None, edge_type_count, neighbor_samples])
        
        # Look up embeddings for nodes
        if feature_dic is not None:
            node_embed = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = tf.matmul(node_embed, embed_trans)
        else:
            node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        
        if feature_dic is not None:
            node_embed_neighbors = tf.nn.embedding_lookup(node_features, node_neigh)
            node_embed_tmp = tf.concat([tf.matmul(tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, feature_dim]), tf.reshape(tf.slice(u_embed_trans, [i, 0, 0], [1, -1, -1]), [feature_dim, embedding_u_size])) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(tf.reshape(node_embed_tmp, [edge_type_count, -1, neighbor_samples, embedding_u_size]), axis=2), perm=[1,0,2])
        else:
            node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
            node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]), [1, -1, neighbor_samples, embedding_u_size]) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2])

        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)
        
        attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
        node_type_embed = tf.matmul(attention, node_type_embed)
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])

        if feature_dic is not None:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + tf.matmul(node_feat, feature_weights)

        last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=last_node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    
    
    with tf.Session(graph=graph) as sess:
        all_vec_dict = {} #这个词典保存所有顶点的向量，是final-model中所有向量的求和
        count = 0

        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)


        print('Training')
        g_iter = 0
        best_score = 0
        patience = 0
        for epoch in range(epochs):#每一个epoch都打乱顺序
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)

            data_iter = tqdm.tqdm(batches,
                                desc="epoch %d" % (epoch),
                                total=(len(train_pairs) + (batch_size - 1)) // batch_size,
                                bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in enumerate(data_iter):
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2], node_neigh: data[3]}
                _, loss_value, summary_str = sess.run([optimizer, loss, merged], feed_dict)
                writer.add_summary(summary_str, g_iter)

                g_iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))
            
            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i], node_neigh: [neighbors[j]]})[0])
            #从这里开始检查是不是我要的向量
            print('==========================')

            print('final_model是一个字典。它的键的数量是：')
            print(len(final_model))
            # input("这里暂停一下")

            print('输出final_model的键值，以及该键值下的词典的长度：')
            for kkk in final_model.keys():
                print(kkk)
                print(len(final_model[kkk]))
                for kkkk in final_model[kkk].keys():
                    print('打印一个向量的维度')
                    print(len(final_model[kkk][kkkk]))
                    break
            # input("这里暂停一下")

            print('==========================')

            for _ in final_model['1']:
                if _ not in all_vec_dict.keys():
                    all_vec_dict[_] = final_model['1'][_]
                else:
                    all_vec_dict[_] = all_vec_dict[_] + final_model['1'][_]
            
                count += 1
                print('目前相加了' + str(count) + '次') #30228
            
            
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(final_model[edge_types[i]], valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr, y_scores, y_true = evaluate(final_model[edge_types[i]], testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]], output=True)
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print('valid auc:', np.mean(valid_aucs))
            print('valid pr:', np.mean(valid_prs))
            print('valid f1:', np.mean(valid_f1s))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print('Early Stopping')
                    break
            
                
    return average_auc, average_f1, average_pr, all_vec_dict, y_scores, y_true

   
if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    if args.features is not None:
        feature_dic = {}
        with open(args.features, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                items = line.strip().split()
                feature_dic[items[0]] = items[1:]
    else:
        feature_dic = None

    log_name = file_name.split('/')[-1] + f'_evaltype_{args.eval_type}_b_{args.batch_size}_e_{args.epoch}'

    training_data_by_type = load_training_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    average_auc, average_f1, average_pr, all_vec_dict, y_scores, y_true = train_model(training_data_by_type, feature_dic, log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())))

    print('Overall ROC-AUC:', average_auc)
    print('Overall PR-AUC', average_pr)
    print('Overall F1:', average_f1)

    #下面开始把向量拿来计算分数
    #首先把第一列提取出来，这是名字
            
    all_names = []
    for kkk in all_vec_dict.keys():
        all_names.append(kkk)
    print('all_vec_dict中一共有' + str(len(all_names)) + '个key')
    input("这里暂停一下")

    #把这一列的每一个元素拆分成列表，然后看开头是B还是C，用来区分药物和蛋白质，然后把文件分成两个部分。一个是药物，一个是蛋白质
    drug_name, target_name = [], []
    for _ in all_names:
        if _[0] == 'D':
            drug_name.append(_)
        elif _[0] == 'C':
            target_name.append(_)

    print('药物的数量：' + str(len(drug_name)))
    print('蛋白的数量：' + str(len(target_name)))
    print('打印药物前10个：', drug_name[:10])
    print('打印药物前10个：', target_name[:10])

    input("这里暂停一下")
    #两两取出数据进行比较，用代码utilis中的计算方法去计算分数。然后保存在一个列表中
    final_scores = []
    for d in drug_name:
        for t in target_name:
            score = np.dot(all_vec_dict[d], all_vec_dict[t]) / (np.linalg.norm(all_vec_dict[d]) * np.linalg.norm(all_vec_dict[t]))
            final_score = [d, t, score]
            final_scores.append(final_score)
    print('检查final_scores总共有多少行：' + str(len(final_scores)))
    print("下面开始检查finnal_scores的形式")
    print("final_scores里面的第一行的形式", len(final_scores[0]))
    print("final_scores里面的第一行的形式", final_scores[0])

    input("这里暂停一下")

    #最后把列表输出
    with open('./result_data/final_all_score.txt', 'w', encoding='utf-8') as output:
        for row in final_scores:
            rowtext = '{} {} {}'.format(row[0], row[1], row[2])
            output.write(rowtext)
            output.write('\n')

    #输出画auc的分数_真值表
    zipped = list(zip(y_scores, y_true))

    with open('./result_data/true_data_score.txt', 'w', encoding='utf-8') as output:
        for row in zipped:
            rowtext = '{} {}'.format(row[0], row[1])
            output.write(rowtext)
            output.write('\n')