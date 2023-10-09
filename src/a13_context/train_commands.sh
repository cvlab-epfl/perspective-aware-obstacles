
cd /cvlabdata2/home/lis/dev/unknown-dangers
cmd_train="python -m src.a12_inpainting.discrepancy_experiments train "
cmd_eval="python -m src.a12_inpainting.discrepancy_experiments evaluation --no-perframe --no-islands "
cmd_table="python -m src.a12_inpainting.metrics --table "
dsets="FishyLAF-LafRoi,RoadObstacles2048p-full"

# $cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_l1
# 	# slow but going
# 	# old setup of 4 inter channels instead of 6

# $cmd_train LapNet_RN50fr_Ker21_learn-1ch-initlap_l1
# 	# 
# 	# old setup of 4 inter channels instead of 6


# $cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_learn-abs
# 	# NAN
# 	# the learned average was not initialized to 1/n, now fixed

# $cmd_train LapNet_RN50fr_Ker21_learn-1ch-initlap_learn-abs
# 	# does not improve during training
# 	# needs restart after the above change to init to 1/n


# # lets try bilinear instead of upconv

# $cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_l1_UPbilinear
# $cmd_train LapNet_RN50fr_Ker21_learn-1ch-initlap_l1_UPbilinear

# # lets try 3x3 mix

# $cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_l1_UPbilinear_mix3
# $cmd_train LapNet_RN50fr_Ker21_learn-1ch-initlap_l1_UPbilinear_mix3

# # lets try weighted avg

# #$cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_l1_UPbilinear
# #$cmd_train LapNet_RN50fr_Ker21_learn-1ch-initlap_l1_UPbilinear


# trainable backbone

exps_unfrozen="LapNet_RN50_Ker21_fixed-laplacian_l1_UPbilinear,LapNet_RN50_Ker21_fixed-laplacian_learn-abs_UPbilinear,LapNet_RN50_Ker21_1ch-initlap_l1_UPbilinear,LapNet_RN50_Ker21_1ch-initlap_learn-abs_UPbilinear"

$cmd_train LapNet_RN50_Ker21_fixed-laplacian_l1_UPbilinear &
$cmd_train LapNet_RN50_Ker21_fixed-laplacian_learn-abs_UPbilinear &
$cmd_train LapNet_RN50_Ker21_1ch-initlap_learn-abs_UPbilinear &
$cmd_train LapNet_RN50_Ker21_1ch-initlap_l1_UPbilinear &
wait
$cmd_eval $exps_unfrozen FishyLAF-LafRoi --comparison LapNet02_unfrozen &
$cmd_eval $exps_unfrozen RoadObstacles2048p-full --comparison LapNet02_unfrozen &
wait



exps_frozen="LapNet_RN50fr_Ker21_fixed-laplacian_l1_UPbilinear,LapNet_RN50fr_Ker21_fixed-laplacian_learn-abs_UPbilinear,LapNet_RN50fr_Ker21_1ch-initlap_l1_UPbilinear"

$cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_l1_UPbilinear &
$cmd_train LapNet_RN50fr_Ker21_fixed-laplacian_learn-abs_UPbilinear &
$cmd_train LapNet_RN50fr_Ker21_1ch-initlap_l1_UPbilinear &
wait
$cmd_eval $exps_frozen FishyLAF-LafRoi  --comparison LapNet03_frozen &
$cmd_eval $exps_frozen RoadObstacles2048p-full --comparison LapNet03_frozen &
wait

$cmd_table $exps_frozen,$exps_unfrozen,ImgVsInp-archResy-NoiseImg-trainFusionv2blur5 $dsets --comparison LapNet03


# $cmd_table $exps_unfrozen,ImgVsInp-archResy-NoiseImg-trainFusionv2blur5 $dsets --comparison LapNet03



# $cmd_eval LapNet_RN50fr_Ker21_fixed-laplacian_l1_UPbilinear,LapNet_RN50fr_Ker21_learn-1ch-initlap_l1_UPbilinear,LapNet_RN50fr_Ker21_fixed-laplacian_l1_UPbilinear_mix3,LapNet_RN50fr_Ker21_learn-1ch-initlap_l1_UPbilinear_mix3,LapNet_RN50fr_Ker21_fixed-laplacian_learn-abs,LapNet_RN50fr_Ker21_fixed-laplacian_learn-abs FishyLAF-LafRoi,RoadObstacles2048p-full --comparison LapNet01



# LapNet_RN50_Ker21_fixed-laplacian_l1
# LapNet_RN50_Ker21_learn-1ch-initlap_l1

# LapNet_RN50_Ker21_fixed-laplacian_learn-abs
# LapNet_RN50_Ker21_learn-1ch-initlap_learn-abs