# Compress the LAF dataset from original 2048x1024 PNG to 1024x512 WEBP
# $DIR_LAF = location of original dataset
# $DIR_LAF_SMALL = output location


DIR_LAF=/cvlabsrc1/cvlab/dataset_LostAndFound/2048x1024_png
DIR_LAF_WEBP=/cvlabsrc1/cvlab/dataset_LostAndFound/2048x1024_webp

python compress_images.py \
	$DIR_LAF/leftImg8bit \
	$DIR_LAF_WEBP/leftImg8bit \
	"cwebp {src} -o {dest} -q 90 -sharp_yuv -m 6" \
	--ext ".webp" --concurrent 20

