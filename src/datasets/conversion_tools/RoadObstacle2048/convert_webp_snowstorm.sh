
DIR_R=/mnt/data-research/datasets/dataset_RoadObstacleWeather_v1/

python compress_and_rename.py \
	$DIR_R/images \
	$DIR_R/images_Webp \
	"cwebp {src} -o {dest} -q 85 -sharp_yuv -m 6" \
	--ext ".webp" --concurrent 20
