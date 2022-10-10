import os
import tqdm
import scipy
import argparse
import numpy as np
import scipy.io as scio
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils.Update import Inference, H_Loss
from utils.logger import setup_logger, mkdir
from utils.dataloader import data_iterator_T, data_iterator
from utils.metric import eval_map, one_hot_label_db

def parse_args():
    """parse the model arguments"""
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--num_gpus', type=int, default='0', help='Number of GPU')
    parser_arg.add_argument('--version', type=str, default='debug', help='Record the version of this times')
    parser_arg.add_argument('--codelen', type=int, default=32, help='Length of binary code expected to be generated')
    parser_arg.add_argument('--batch_size', type=int, default=15, help='batch size')

    parser_arg.add_argument('--round', type=int, default=10, help='How many round of the model iteration')
    parser_arg.add_argument('--max_epoch', type=int, default=10, help='Maximum epoch number for each round')
    parser_arg.add_argument('--lr', type=float, default=0.0002, help='Learning rate')

    parser_arg.add_argument('--m1', type=int, default=1000, help='m1 of fine-grained components number')
    parser_arg.add_argument('--m2', type=int, default=100, help='m2 of coarse-grained components number')

    parser_arg.add_argument('--lambd', type=float, default=0, help='Coefficient for components correlation')

    return parser_arg.parse_args()

version = parse_args().version
num_gpus = parse_args().num_gpus
codelen = parse_args().codelen
batch_size = parse_args().batch_size
learning_rate = parse_args().lr
round = parse_args().round
max_epoch = parse_args().max_epoch
m1 = parse_args().m1
m2 = parse_args().m2
lambd = parse_args().lambd

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

args = parse_args()

# saving path
data_path = '/apdcephfs/share_1367250/qinghonglin/hash_datasets/cifar-10/cifar-10.mat'
save_path = '/apdcephfs/private_qinghonglin/hash_DSCH/checkpoints/output/cifar-10/' + version
logfile_dir = '/apdcephfs/private_qinghonglin/hash_DSCH/checkpoints/log/cifar-10/'
logfile_name = version + '.log'
mkdir(save_path)

logger = setup_logger(logfile_name, logs_dir=logfile_dir, also_stdout=False)
logger.info('dataset: cifar-10')
for arg in vars(args):
    print(arg, getattr(args, arg))
    logger.info(arg + ' : '+ str(getattr(args, arg)))
print()
logger.info('')

# data preparation
print('0. loading the dataset')
mat = scio.loadmat(data_path)
if args.version == 'debug':
    train_data, train_data_L = mat['test_data'], mat['test_L']
else:
    train_data, train_data_L = mat['train_data'], mat['train_L']

num_of_train = train_data.shape[-1]
total_batch = int(np.floor(num_of_train / batch_size))
img224, img64 = [], []

for i in tqdm.tqdm(range(num_of_train)):
    t = train_data[:, :, :, i]
    if train_data.shape[0] != 224:
        image = scipy.misc.imresize(t, [224, 224])
        img224.append(image)
    else:
        img224.append(t)
    img2 = scipy.misc.imresize(t, [64, 64]).astype(np.float32)
    img64.append(img2 / 255.0)
img224 = np.array(img224)
img64 = np.array(img64)
del train_data

# Construct the graph
graph = tf.Graph()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with graph.as_default(), tf.device('/gpu:' + str(num_gpus)):
    tower_grads = []
    train_mode = tf.compat.v1.placeholder(tf.bool)
    lambd_s = tf.compat.v1.placeholder(tf.float32, shape=[])
    PI_1_s = tf.compat.v1.placeholder(tf.float32, [None, None])
    PI_2_s = tf.compat.v1.placeholder(tf.float32, [None, None])
    MU_1_s = tf.compat.v1.placeholder(tf.float32, [None, codelen])
    MU_2_s = tf.compat.v1.placeholder(tf.float32, [None, codelen])
    S_s = tf.compat.v1.placeholder(tf.float32, [None, None])
    P_1_s = tf.compat.v1.placeholder(tf.float32, [None, m1])
    P_2_s = tf.compat.v1.placeholder(tf.float32, [None, m2])

    input = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3], name='input')

    learning_rate_s = tf.compat.v1.placeholder(tf.float32, shape=[])
    global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate_decay = tf.compat.v1.train.exponential_decay(learning_rate_s, global_step, 200, 0.99, staircase=True)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_decay, epsilon=1.0)

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
        with tf.compat.v1.device('/gpu:' + str(num_gpus)):
            with tf.name_scope('tower_0') as scope:
                H_code, sim1, sim2, sim3 = Inference(input, train_mode=train_mode, codelen=codelen,
                                                               components_F=MU_1_s, components_C=MU_2_s)
                loss_ic, loss_cc_f, loss_cc_c = \
                    H_Loss(S=S_s, P1=P_1_s, P2=P_2_s,
                           sim1=sim1, sim2=sim2, sim3=sim3,
                           pi1=PI_1_s, pi2=PI_2_s)
                loss = loss_ic + lambd_s * loss_cc_f + lambd_s * loss_cc_c

                grads = optimizer.compute_gradients(loss)
                train_g = optimizer.apply_gradients(grads, global_step=global_step)

    sess = tf.compat.v1.InteractiveSession(graph=graph, config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

def Save_binary_code(data_path, save_path, batch_size):
    mat = scio.loadmat(data_path)
    dataset = mat['data_set']
    test = mat['test_data']
    del mat

    dataset_images = []  # This is for dataset
    test_images = []
    dataset_ = []
    train_ = []
    test_ = []

    for i in tqdm.tqdm(range(dataset.shape[-1])):
        t = dataset[:, :, :, i]
        if dataset.shape[0] != 224:
            image = scipy.misc.imresize(t, [224, 224])
            dataset_images.append(image)
        else:
            dataset_images.append(t)
    dataset_images = np.array(dataset_images)
    del dataset

    for i in tqdm.tqdm(range(test.shape[-1])):
        t = test[:, :, :, i]
        if test.shape[0] != 224:
            image = scipy.misc.imresize(t, [224, 224])
            test_images.append(image)
        else:
            test_images.append(t)
    test_images = np.array(test_images)
    del test

    print('generate binary code for dataset...')
    logger.info('generate binary code for dataset...')
    for i in tqdm.tqdm(range(0, len(dataset_images), batch_size)):
        batch_image = dataset_images[i:i + batch_size]
        H_all = sess.run(H_code, feed_dict={input: batch_image, train_mode: False})
        if i == 0:
            dataset_ = H_all
        else:
            dataset_ = np.concatenate((dataset_, H_all), axis=0)
    del dataset_images

    print('generate binary code for testset...')
    logger.info('generate binary code for testset...')
    for i in tqdm.tqdm(range(0, len(test_images), batch_size)):
        batch_image = test_images[i:i + batch_size]
        H_test = sess.run(H_code, feed_dict={input: batch_image, train_mode: False})
        if i == 0:
            test_ = H_test
        else:
            test_ = np.concatenate((test_, H_test), axis=0)
    del test_images

    print('save code...')
    logger.info('save code...')
    np.savez(save_path + '.npz', dataset=dataset_, train=train_, test=test_)

max_map = 0
max_round = 0

# 1.Train Encoder parameters
for r in range(round):
    epoch = 0

    while epoch < max_epoch:
        # Update network
        print('Training the network...')
        print("Epoch: {0}".format(epoch))
        mean_acc = []
        mean_loss = []
        mean_loss_ic = []
        mean_loss_cc_f = []
        mean_loss_cc_c = []
        iter_ = data_iterator(img224, batch_size)
        iter_T = data_iterator_T(img224, batch_size)

        for i in tqdm.tqdm(range(0, len(img224), batch_size)):
            next_img224 = img224[i:i + batch_size]
            H_code_ = sess.run(H_code, feed_dict={input: next_img224, train_mode: False})
            if i == 0:
                H_train_ = H_code_
            else:
                H_train_ = np.concatenate((H_train_, H_code_), axis=0)

        # Generate fine-grained components
        gmm = GaussianMixture(n_components=m1)
        gmm.fit(H_train_)

        P_1_ = gmm.predict_proba(H_train_)    # Assignment of fine-grained components: p^1_{ji} of Eq (11)
        MU_1_ = gmm.means_      # Representation of fine-grained components: \mu^1_{j} of Eq (10)
        PI_1_ = gmm.weights_    # Prior probability of fine-grained components: \pi^1_{j} of Eq (10)
        PI_1_ = np.expand_dims(PI_1_, 0).repeat(2 * batch_size, axis=0) # double for data augmentation
        S_ = np.matmul(P_1_, P_1_.T)            # s_{ij} of Eq (15)

        # Generate coarse-grained components
        kmean = KMeans(n_clusters=m2)
        kmean.fit(MU_1_)

        P_21_ = one_hot_label_db(kmean.labels_) # Assignment of fine-grained components to coarse-grained: p(C^2|C^1) of Eq (12)
        MU_2_ = kmean.cluster_centers_          # Representation of coarse-grained components: \mu^2_{j} of Eq (13)
        PI_2_ = np.ones((m2, )) / m2            # Prior probability of coarse-grained components equal to 1/m
        PI_2_ = np.expand_dims(PI_2_, 0).repeat(2 * batch_size, axis=0)    # double for data augmentation
        P_2_ = np.matmul(P_1_, P_21_)           # Assignment of coarse-grained components: p^2_{ki} of Eq (14)

        # double for data augmentation
        # S_ = np.concatenate((S_, S_), axis=0)
        # S_ = np.concatenate((S_, S_), axis=1)
        P_1_ = np.concatenate((P_1_, P_1_), axis=0)
        P_2_ = np.concatenate((P_2_, P_2_), axis=0)

        S_ = np.identity(len(img224))
        S_ = np.concatenate((S_, S_), axis=0)
        S_ = np.concatenate((S_, S_), axis=1)

        for i in tqdm.tqdm(range(total_batch)):
            next_batch, next_idx = iter_T.__next__()
            S_batch = S_[next_idx, :][:, next_idx]
            P_1_batch = P_1_[next_idx, :]
            P_2_batch = P_2_[next_idx, :]

            loss_, loss_ic_, loss_cc_f_, loss_cc_c_, _ = \
                sess.run([loss, loss_ic, loss_cc_f, loss_cc_c, train_g],
                         feed_dict={input: next_batch, lambd_s: lambd,
                                    S_s: S_batch,
                                    MU_1_s: MU_1_, P_1_s: P_1_batch,
                                    MU_2_s: MU_2_,  P_2_s: P_2_batch,
                                    PI_1_s: PI_1_, PI_2_s: PI_2_,
                                    learning_rate_s: learning_rate, train_mode: True})

            mean_loss.append(loss_)
            mean_loss_ic.append(loss_ic_)
            mean_loss_cc_f.append(lambd * loss_cc_f_)
            mean_loss_cc_c.append(lambd * loss_cc_c_)

            print("total_mean_loss:{:.3f}, mean_loss_is:{:.3f}, mean_loss_cc_f:{:.3f}, mean_loss_cc_c:{:.3f}".format(
                np.mean(mean_loss), np.mean(mean_loss_ic), np.mean(mean_loss_cc_f),  np.mean(mean_loss_cc_c)))

        logger.info("Round: {0}".format(r))
        logger.info("mean_loss: %f" % np.mean(mean_loss))
        logger.info("mean_loss_is: %f" % np.mean(mean_loss_ic))
        logger.info("mean_loss_cc_f: %f" % np.mean(mean_loss_cc_f))
        logger.info("mean_loss_cc_c: %f" % np.mean(mean_loss_cc_c))
        logger.info('')

        epoch += 1

    # 2.Saving the model
    print('Saving the Model of Round:', r)
    temp_save_path = save_path + '/' + str(r) + '_epoch_' + str(max_epoch)
    Save_binary_code(data_path, temp_save_path, 50)

    # 3.Evaluating
    print('Evaluating...')
    map_1000, map_5000, map_all = eval_map(temp_save_path, dataset='cifar')
    print('codelen =', codelen, ', map:{:.4f}, map_1000:{:.4f}, map_5000:{:.4f}'.
          format(map_all, map_1000, map_5000))
    logger.info('codelen:' + str(codelen) + ' map:{:.4f}, map_1000:{:.4f}, map_5000:{:.4f}'.
                format(map_all, map_1000, map_5000))

    if map_5000 > max_map:
        max_map = map_5000
        max_round = r

print('codelen =' + str(codelen) +  ', max_map_5000:{:.4f} with round {:0}'.format(max_map, max_round))
logger.info('codelen =' + str(codelen) + ', max_map_5000:{:.4f} with round {:0}'.format(max_map, max_round))