# Compress the Cityscapes dataset from original 2048x1024 PNG to 1024x512 WEBP
# $DIR_CITYSCAPES = location of original dataset
# $DIR_CITYSCAPES_SMALL = output location

#DIR_CITYSCAPES=/cvlabdata1/cvlab/dataset_cityscapes
#DIR_CITYSCAPES_SMALL=/cvlabsrc1/cvlab/dataset_Cityscapes/1024x512_webp

DIR_IDD=$MY_DIR_DATASETS/dataset_IDD_IndiaDrivingDataset/idd20k_segmentation

python compress_images.py \
	$DIR_IDD/leftImg8bit \
	$DIR_IDD/leftImg8bit_webp \
	"cwebp {src} -o {dest} -q 90 -sharp_yuv -m 6" \
	--ext ".webp" --concurrent 20

# python compress_images.py \
# 	$DIR_CITYSCAPES/gtFine \
# 	$DIR_CITYSCAPES_SMALL/gtFine \
# 	"convert {src} -filter point -resize 50% {dest}" \
# 	--ext ".png" --concurrent 20
