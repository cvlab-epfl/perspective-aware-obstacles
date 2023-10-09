
# generate set of crops
# store the id of the class in the center, to choose crops for cri-split

python -m src.a10_geometric_instances.generate_crops_indexed \
	--dir_in $DIR_DATASETS/dataset_Cityscapes/2048x1024 \
	--dir_out $DIR_DATA/datasets_processed/Cityscapes2048__SpatialEmbeddings_object-crops-512 \
	--classes all_objects \
	--split train \
	--ext_in .webp \


python -m src.a10_geometric_instances.generate_crops_indexed \
	--dir_in $DIR_DATASETS/dataset_Cityscapes/2048x1024 \
	--dir_out $DIR_DATA/datasets_processed/Cityscapes2048__SpatialEmbeddings_object-crops-512 \
	--classes all_objects \
	--ext_in .webp \
	--split val
