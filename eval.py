from utils.metric import eval_map, one_hot_label_db

r = 0
codelen = 16
temp_save_path = "/apdcephfs/private_qinghonglin/hash_DSCH/checkpoints/output/cifar-10/16_01_m1_1000_01_m2_500_10_q/0_epoch_10"

# 3.Evaluating
print('Evaluating...')
map_1000, map_5000, map_all = eval_map(temp_save_path, dataset='cifar')
print('Round =', r, 'codelen =', codelen, ', map:{:.4f}, map_1000:{:.4f}, map_5000:{:.4f}'.
      format(map_all, map_1000, map_5000))