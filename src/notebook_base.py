
from IPython import get_ipython
ipython = get_ipython()

ipython.run_cell("""
%load_ext autoreload
%aimport -numpy -cv2 -torch -matplotlib -matplotlib.pyplot
%autoreload 2
""")

from src.common.util_notebook import *
if not globals().get('PLT_STYLE_OVERRIDE'):
	plt.style.use('dark_background')
else:
	print('No plt style set')
plt.rcParams['figure.figsize'] = (12, 8)
ipython.run_cell("""
%matplotlib inline
%config InlineBackend.figure_format = 'jpg'
""")

from src.pytorch_selection import *
pytorch_init()

from src.paths import *
from src.pipeline.frame import *
from src.pipeline.config import *
from src.pipeline.transforms import *
from src.pipeline.transforms_pytorch import *
from src.pipeline.transforms_imgproc import *
from src.pipeline.pipeline import *
from src.pipeline.experiment import *

from src.datasets.dataset import *
from src.datasets.cityscapes import *
from src.datasets.lost_and_found import *
from src.datasets.road_anomaly import *
from src.datasets.NYU_depth_v2 import *


from src.a01_sem_seg.networks import *
from src.a01_sem_seg.transforms import *
from src.a01_sem_seg.experiments import *
from src.a01_sem_seg.class_statistics import *

from src.a04_reconstruction.networks import *
from src.a04_reconstruction.transforms import *
from src.a04_reconstruction.experiments import *

from src.a05_differences.networks import *
from src.a05_differences.transforms import *
from src.a05_differences.experiments import *
from src.a05_differences.experiments_nyu import *
from src.a05_differences.metrics import *

from src.a05_road_rec_baseline.networks import *
from src.a05_road_rec_baseline.transforms import *
from src.a05_road_rec_baseline.experiments import *
