
DIR_B=/mnt/data-research-2/data_research_tmp/

# python process_dset.py \
# 	$DIR_B/labeling \
# 	$DIR_B/dataset_RoadObstacle_1920 \
# 	--method 1920

python process_dset.py \
	$DIR_B/labeling \
	$DIR_B/dataset_RoadObstacle_1920_flat \
	--method 1920 --flat

