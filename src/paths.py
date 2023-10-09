
import os
from pathlib import Path

DIR_DATA = Path(os.environ.get('DIR_DATA', '/cvlabdata2/home/lis/data'))
DIR_DATA_cv1 = Path(os.environ.get('DIR_DATA_cv1', '/cvlabdata1/home/lis/data'))

DIR_EXP = Path(os.environ.get('DIR_EXPERIMENTS', '/cvlabdata2/home/lis/exp'))

DIR_EXP2 = Path(os.environ.get('DIR_EXPERIMENTS', '/cvlabdata2/home/lis/exp2'))


DIR_DSETS = Path(os.environ.get('DIR_DATASETS', '/cvlabsrc1/cvlab')) #/labirynth/data/datasets
# DIR_DSETS2 = Path(os.environ.get('MY_DIR_DATASETS2', '/cvlabsrc1/cvlab'))
DIR_DSETS_SMALL = Path(os.environ.get('MY_DIR_DATASETS_SMALL', DIR_DSETS))

DIR_BDD_SEG = Path(os.environ.get('MY_DIR_BDD_SEG', DIR_DSETS / 'dataset_BDD100k/bdd100k/seg'))
DIR_APOLLO = Path(os.environ.get('MY_DIR_APOLLO', DIR_DSETS / 'dataset_ApolloScape/scenes'))
DIR_AUTONUE = Path(os.environ.get('MY_DIR_AUTONUE', DIR_DSETS / 'dataset_AutoNUE/anue/'))
