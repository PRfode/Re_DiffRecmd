# debug_reader.py
import argparse, logging, sys
sys.path.append('..')                 # 指向 ReChorus 根目录
from helpers.SeqReader import SeqReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')

parser = argparse.ArgumentParser()
parser = SeqReader.parse_data_args(parser)
args = parser.parse_args(['--path', 'data', '--dataset', 'MovieLens_1M/ML_1MGDR', '--sep', ','])
reader = SeqReader(args)

print('n_users:', reader.n_users,
      'n_items:', reader.n_items,
      'train shape:', reader.data_df['train'].shape)
print('user_his[89]:', reader.user_his.get(89, 'N/A'))
print('train.head():\n', reader.data_df['train'].head())