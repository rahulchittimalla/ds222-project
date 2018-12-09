from __future__ import division

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys
import ctypes
# import threading

parameter_servers = ["10.24.1.218:2223"]
workers = ["10.24.1.219:2224", "10.24.1.220:2225"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

export_path = "/home/swapnilgupta.229/project/JointNRE/data/"

word_vec = np.load(export_path + 'vec.npy')
f = open(export_path + "config", 'r')
config = json.loads(f.read())
f.close()

ll = ctypes.cdll.LoadLibrary #
lib = ll("./init.so")
lib.setInPath("/home/swapnilgupta.229/project/JointNRE/data/")
lib.init()

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True
CONFIG.allow_soft_placement = True
CONFIG.log_device_placement = True

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

tf.app.flags.DEFINE_float('nbatch_kg',100,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_integer('ent_total',lib.getEntityTotal(),'total of entities')
tf.app.flags.DEFINE_integer('rel_total',lib.getRelationTotal(),'total of relations')
tf.app.flags.DEFINE_integer('tri_total',lib.getTripleTotal(),'total of triples')
tf.app.flags.DEFINE_integer('katt_flag', 1, '1 for katt, 0 for att')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', config['textual_rel_total'],'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch',12,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size',160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'learning rate for nn')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')

tf.app.flags.DEFINE_string('model_dir','./model_async/','path to store model')
#tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')


LOG_DIR = 'async_logs'

def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

def make_shape(array,last_dim):
	output = []
	for i in array:
		for j in i:
			output.append(j)
	output = np.array(output)
	if np.shape(output)[-1]==last_dim:
		return output
	else:
		print 'Make Shape Error!'

def get_session(sess):
	session = sess
	while type(session).__name__ != 'Session':
		session = session._sess
	return session

def main(_):

	print 'reading word embedding'
	word_vec = np.load(export_path + 'vec.npy')
	print 'reading training data'

	train_label_total = np.load(export_path + 'train_label.npy')	

	if FLAGS.job_name=="worker" and FLAGS.task_index == 0:
                instance_triple = np.load(export_path + 'train_instance_triple1.npy')
                instance_scope = np.load(export_path + 'train_instance_scope1.npy')
                train_len = np.load(export_path + 'train_len1.npy')
                train_label = np.load(export_path + 'train_label1.npy')
                train_word = np.load(export_path + 'train_word1.npy')
                train_pos1 = np.load(export_path + 'train_pos11.npy')
                train_pos2 = np.load(export_path + 'train_pos21.npy')
                train_mask = np.load(export_path + 'train_mask1.npy')
                train_head = np.load(export_path + 'train_head1.npy')
                train_tail = np.load(export_path + 'train_tail1.npy')
                print 'reading finished'
                print 'mentions                 : %d' % (len(instance_triple))
                print 'sentences                : %d' % (len(train_len))
                print 'relations                : %d' % (FLAGS.num_classes)
                print 'word size                : %d' % (len(word_vec[0]))
                print 'position size    : %d' % (FLAGS.pos_size)
                print 'hidden size              : %d' % (FLAGS.hidden_size)
	
	if FLAGS.job_name=="worker" and FLAGS.task_index == 1:
                instance_triple = np.load(export_path + 'train_instance_triple2.npy')
                instance_scope = np.load(export_path + 'train_instance_scope2.npy')
                train_len = np.load(export_path + 'train_len2.npy')
                train_label = np.load(export_path + 'train_label2.npy')
                train_word = np.load(export_path + 'train_word2.npy')
                train_pos1 = np.load(export_path + 'train_pos12.npy')
                train_pos2 = np.load(export_path + 'train_pos22.npy')
                train_mask = np.load(export_path + 'train_mask2.npy')
                train_head = np.load(export_path + 'train_head2.npy')
                train_tail = np.load(export_path + 'train_tail2.npy')

                print 'reading finished'
                print 'mentions                 : %d' % (len(instance_triple))
                print 'sentences                : %d' % (len(train_len))
                print 'relations                : %d' % (FLAGS.num_classes)
                print 'word size                : %d' % (len(word_vec[0]))
                print 'position size    : %d' % (FLAGS.pos_size)
                print 'hidden size              : %d' % (FLAGS.hidden_size)

	
	reltot = {}
	for index, i in enumerate(train_label_total):
		if not i in reltot:
			reltot[i] = 1.0
		else:
			reltot[i] += 1.0
	for i in reltot:
		reltot[i] = 1/(reltot[i] ** (0.05))

	print(time.time())
	if FLAGS.job_name == "ps":
		server.join()
	elif FLAGS.job_name == "worker":
		Start = time.time()
                as_time_dist_w1 = []
                as_time_dist_w2 = []
                as_save_cost_w1 = []
                as_save_cost_w2 = []
                local_step = 0

		with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
			print 'building network...'
			# sess = tf.Session()
			if FLAGS.model.lower() == "cnn":
				model = network.CNN(is_training = True, word_embeddings = word_vec)
			elif FLAGS.model.lower() == "pcnn":
				model = network.PCNN(is_training = True, word_embeddings = word_vec)
			elif FLAGS.model.lower() == "lstm":
				model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
			elif FLAGS.model.lower() == "gru":
				model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
			elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
				model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
			elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
				model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
			
			global_step = tf.Variable(0,name='global_step',trainable=False)
			#global_step_kg = tf.Variable(0,name='global_step_kg',trainable=False)
			#tf.summary.scalar('learning_rate', FLAGS.learning_rate)
			#tf.summary.scalar('learning_rate_kg', FLAGS.learning_rate_kg)
			
			optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
                        optimizer = tf.contrib.opt.DropStaleGradientOptimizer(optimizer,staleness=10,use_locking=True)
                        grads_and_vars = optimizer.compute_gradients(model.loss + 0.01* model.loss_kg)
                        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
			#merged_summary = tf.summary.merge_all()
			init_op = tf.global_variables_initializer()
			# sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(max_to_keep=None)
			print 'building finished'

			def train(sess, local_step):

				batch_size_kg = (int)(FLAGS.ent_total / FLAGS.nbatch_kg)
				ph = np.zeros(batch_size_kg, dtype = np.int32)
				pt = np.zeros(batch_size_kg, dtype = np.int32)
				pr = np.zeros(batch_size_kg, dtype = np.int32)
				nh = np.zeros(batch_size_kg, dtype = np.int32)
				nt = np.zeros(batch_size_kg, dtype = np.int32)
				nr = np.zeros(batch_size_kg, dtype = np.int32)
				ph_addr = ph.__array_interface__['data'][0]
				pt_addr = pt.__array_interface__['data'][0]
				pr_addr = pr.__array_interface__['data'][0]
				nh_addr = nh.__array_interface__['data'][0]
				nt_addr = nt.__array_interface__['data'][0]
				nr_addr = nr.__array_interface__['data'][0]
				lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


				def train_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope, weights, pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):


					feed_dict = {
                                                model.head_index: head,
                                                model.tail_index: tail,
                                                model.word: word,
                                                model.pos1: pos1,
                                                model.pos2: pos2,
                                                model.mask: mask,
                                                model.len : leng,
                                                model.label_index: label_index,
                                                model.label: label,
                                                model.scope: scope,
                                                model.keep_prob: FLAGS.keep_prob,
                                                model.weights: weights,
                                                model.pos_h: pos_h_batch,
                                                model.pos_t: pos_t_batch,
                                                model.pos_r: pos_r_batch,
                                                model.neg_h: neg_h_batch,
                                                model.neg_t: neg_t_batch,
                                                model.neg_r: neg_r_batch
                                        }
                                        _,step, loss_kg, loss_nn, output, correct_predictions = sess.run([train_op, global_step, model.loss_kg, model.loss, model.output, model.correct_predictions], feed_dict)

					#summary_writer.add_summary(summary, step)
					return output, loss_kg, loss_nn, correct_predictions

				stack_output = []
				stack_label = []
				stack_ce_loss = []

				train_order = range(len(instance_triple))

				save_epoch = 4
				eval_step = 300

				for one_epoch in range(FLAGS.max_epoch):

					print('epoch '+str(one_epoch+1)+' starts!')
					np.random.shuffle(train_order)
					s1 = 0.0
					s2 = 0.0
					tot1 = 0.0
					tot2 = 0.0
					losstot = 0.0


					for i in range(int(len(train_order)/float(FLAGS.batch_size))):
						lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, batch_size_kg)
						#res = train_step_kg(ph, pt, pr, nh, nt, nr)
						time_str = datetime.datetime.now().isoformat()
						#print "epoch %d time %s | loss : %f" % (i, time_str, res)
						### First we are doing a complete epoch of the Knowledge Graph
						#res = 0.0
						#for batch_kg in range(int(FLAGS.nbatch_kg)):
						#	lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, batch_size_kg)
						#	res += train_step_kg(ph, pt, pr, nh, nt, nr)
						#time_str = datetime.datetime.now().isoformat()
						#print "epoch %d time %s | loss : %f" % (i, time_str, res)

						input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
						index = []
						scope = [0]
						label = []
						weights = []
						for num in input_scope:
							index = index + range(num[0], num[1] + 1)
							label.append(train_label[num[0]])
							#if train_label[num[0]] > 53:
							#	print train_label[num[0]]
							scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
							weights.append(reltot[train_label[num[0]]])
						label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
						label_[np.arange(FLAGS.batch_size), label] = 1

						output, batch_losskg, batch_lossnn, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:], train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights, ph, pt, pr, nh, nt, nr)
						local_step += 1

						num = 0
						s = 0
						losstot += batch_lossnn
						for num in correct_predictions:
						
							tot1 += 1.0
                                                        if num:
                                                                s1 += 1.0
                                                        s = s + 1


					
						time_str = datetime.datetime.now().isoformat()
						current_step = tf.train.global_step(sess, global_step)
						print "epoch %d batch %d time %s |KGloss : %f, NNloss : %f, accuracy: %f, global_step: %d" % (one_epoch, i, time_str, batch_losskg, batch_lossnn, s1 / tot1, current_step)
	

						if FLAGS.task_index == 0:
                                                        as_time_dist_w1.append([local_step,current_step,time.time()-Start])
                                                elif FLAGS.task_index == 1:
                                                        as_time_dist_w2.append([local_step,current_step,time.time()-Start])

					if FLAGS.task_index == 0:
                                                #st_time_dist_w1.append(time.time() - Start)
                                                as_save_cost_w1.append([batch_lossnn,batch_losskg,batch_lossnn + 0.01*batch_losskg,s1/tot1])
                                                np.save("as_time_dist_w1.npy",as_time_dist_w1)
                                                np.save("as_save_cost_w1.npy",as_save_cost_w1)
                                                print(as_time_dist_w1[len(as_time_dist_w1)-1])
                                        elif FLAGS.task_index == 1:
                                                #st_time_dist_w2.append(time.time() - Start)
                                                as_save_cost_w2.append([batch_lossnn,batch_losskg,batch_lossnn + 0.01*batch_losskg,s1/tot1])
                                                np.save("as_time_dist_w2.npy",as_time_dist_w2)
                                                np.save("as_save_cost_w2.npy",as_save_cost_w2)
                                                print(as_time_dist_w2[len(as_time_dist_w2)-1])


					if (one_epoch + 1) % save_epoch == 0:
						print 'epoch '+str(one_epoch+1)+' has finished'
						print 'saving model...'
						path = saver.save(get_session(sess),FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag), global_step=current_step)
						print 'have savde model to '+path
		

		stop_hook = tf.train.StopAtStepHook(last_step = 16000*(FLAGS.max_epoch+2))
		#stop_hook = tf.train.StopAtStepHook(last_step = 3664*FLAGS.max_epoch)		
		#summary_hook = tf.train.SummarySaverHook(save_secs = 600, output_dir = LOG_DIR, summary_op = merged_summary)
		#chief_hook = [summary_hook,stop_hook]
		hook = [stop_hook]

		with tf.train.MonitoredTrainingSession(master = server.target, is_chief=(FLAGS.task_index == 0),
                                                                                        checkpoint_dir = LOG_DIR,save_summaries_steps = None,
                                                                                        save_summaries_secs=None, hooks = hook) as sess:
                        sess.run(init_op)
                        while not sess.should_stop():
                                #summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
                                train(sess,local_step)


if __name__ == "__main__":
	tf.app.run() 
