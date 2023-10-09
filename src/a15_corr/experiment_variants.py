from ..pipeline.config import extend_config


def exp15_configs():
	DEFAULTS = dict(
		discrepancy = dict(
			net = dict(
				batch_train = 12,
				batch_eval = 6,
				batch_infer = 1,

				separate_gen_image = False,
				perspective = False,

				arch_core = 'v1',
				arch_backbone = 'resnext101_32x8d',
				arch_feature_procs = ['pass'],
				arch_classifier = 'last',
			),
			train = dict(
				dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
				mod_sampler = 'v1-768',

				num_workers = 4,
				epoch_limit = 65,

				# loss
				loss_name = 'cross_entropy',
				class_weights = None,
				augmentation_noise = None,
			),
			preproc_blur = False,
		)
	)

	cfgs = []
	cfgs_by_name = {}

	def add(base=DEFAULTS, REPEATS=4, **diff):
		c = extend_config(base, diff)
		basename = c['name']
		cfgs.append(c)
		cfgs_by_name[basename] = c

		if REPEATS and REPEATS > 0:
			for i in range(REPEATS):
				add(base=cfgs_by_name[basename], REPEATS=0, name=f'{basename}-rep{i+1}')

		return c

	def repeat(basename, number=1):
		for i in range(number):
			add(base=cfgs_by_name[basename], REPEATS=0, name=f'{basename}-rep{i+1}')

	add(
		name = '1500-1-CrBaseR101',
	)

	add(
		name = '1500-2-CrBaseR50',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext50_32x4d',
			),
		),
	)

	add(
		name = '1501-1-CrSumSimple',
		discrepancy = dict(
			net = dict(
				arch_classifier = 'sumSimple',
			)	
		),
	)

	add(
		name = '1501-2-CrNeighbourD1',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD1'],
			),
		),
	)

	add(
		name = '1501-3-CrNeighbourD2',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2'],
			),
		),
	)

	add(
		name = '1501-4-CrNeighbourD1Sq',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD1Sq'],
			),
		),
	)

	add(
		name = '1502-1-CrNeighbourD2S',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2'],
				arch_feature_procs_sticky = ['neighboursD2'],
			),
		),
	)

	add(
		name = '1502-2-CrSum_NeighbourD2Stck',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2'],
				arch_feature_procs_sticky = ['neighboursD2'],

				arch_classifier = 'sumSimple',
			),
		),
	)

	add(
		name = '1502-3-CrNeighbourD2Bn',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2Bn'],
			),
		),
	)

	add(
		name = '1502-4-CrNeighbourD2BnStck',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2Bn'],
				arch_feature_procs_sticky = ['neighboursD2Bn'],
			),
		),
	)

	add(
		name = '1502-5-CrCovcat',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursChanVar05'],
			),
		),
	)

	add(
		name = '1502-6-CrCovonly',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['neighboursChanVar05'],
			),
		),
	)

	add(
		name = '1502-6-CrCovonly',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['neighboursChanVar05'],
			),
		),
	)

	add(
		name = '1503-1-CrBaseUpInterp',
		discrepancy = dict(
			net = dict(
				arch_core = 'v2Upinterp',
			)
		)
	)

	add(
		name = '1503-2-CrSumUpInterp',
		discrepancy = dict(
			net = dict(
				arch_core = 'v2Upinterp',
				arch_classifier = 'sumSimple',
			)	
		),
	)

	dset_P3A = dict(train = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp3A_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp3A_cityscapes-val',
	))
	dset_P3B = dict(train = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp3B_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp3B_cityscapes-val',
	))

	add(
		name = '1504-1-P3-Base',
		discrepancy = dict(
			**dset_P3A,
		),
	)

	add(
		name = '1504-2-P3-pwFf',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'perswaveFlatfixed'],
				arch_feature_procs_sticky = ['perswaveFlatfixed'],
			),
			**dset_P3A,
		),
	)

	add(
		name = '1504-3-P3-pwFfModpsm',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'perswaveFlatfixedModpsm'],
				arch_feature_procs_sticky = ['perswaveFlatfixedModpsm'],
			),
			**dset_P3A,
		),
	)

	
	add(
		name = '1504-4-P3B-Base',
		discrepancy = dict(
			**dset_P3B,
		),
	)

	add(
		name = '1504-5-P3B-pwFfModpsm',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'perswaveFlatfixedModpsm'],
				arch_feature_procs_sticky = ['perswaveFlatfixedModpsm'],
			),
			**dset_P3B,
		),
	)

	add(
		name = '1504-6-P3B-pdirect',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			**dset_P3B,
		),
	)

	dset_P3C = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp3C_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp3C_cityscapes-val',
		epoch_limit = 50,

	)

	dset_P3D = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp3D_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp3D_cityscapes-val',
		epoch_limit = 50,
	)

	dset_P3E = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp3E_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp3E_cityscapes-val',
		epoch_limit = 50,
	)

	dset_D2b = dict(
		dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
		dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
		epoch_limit = 50,
	)

	add(
		name = '1504-7-P3C-Base',
		discrepancy = dict(
			train = dict(
				**dset_P3C,
			),
		),
	)

	add(
		name = '1504-8-P3D-Base',
		discrepancy = dict(
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		name = '1504-9-P3C-BaseFocal',
		discrepancy = dict(
			train = dict(
				**dset_P3C,
				loss_name = "focal",
				class_weights = None,
				focal_loss_gamma = 3.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
			),
		),
	)

	add(
		name = '1504-10-P3D-BaseFocal',
		discrepancy = dict(
			train = dict(
				**dset_P3D,
				loss_name = "focal",
				class_weights = None,
				focal_loss_gamma = 3.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
			),
		),
	)

	add(
		name = '1504-11-P3C-CrSumNeighbourD2Stck',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2'],
				arch_feature_procs_sticky = ['neighboursD2'],

				arch_classifier = 'sumSimple',
			),
			train = dict(
				**dset_P3C,
			)
		),
	)

	add(
		name = '1504-12-P3D-CrSumNeighbourD2Stck',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2'],
				arch_feature_procs_sticky = ['neighboursD2'],

				arch_classifier = 'sumSimple',
			),
			train = dict(
				**dset_P3D,
			)
		),
	)


	add(
		name = '1504-12-P3D-CrSumNeighbourD2Stck',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'neighboursD2'],
				arch_feature_procs_sticky = ['neighboursD2'],

				arch_classifier = 'sumSimple',
			),
			train = dict(
				**dset_P3D,
			)
		),
	)

	add(
		name = '1505-1-P3C-BaseBc8',
		discrepancy = dict(
			train = dict(
				class_weights = [1, 8],
				**dset_P3C,
			),
		),
	)

	add(
		name = '1505-2-P3D-BaseBc8',
		discrepancy = dict(
			train = dict(
				class_weights = [1, 8],
				**dset_P3D,
			),
		),
	)

	add(
		name = '1505-3-P3D-PdirectBc8',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				class_weights = [1, 8],
				**dset_P3D,
			),
		),
	)

	add(
		name = '1505-4-P3D-PdirectBc8Sum',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
				arch_classifier = 'sumSimple',
			),
			train = dict(
				class_weights = [1, 8],
				**dset_P3D,
			),
		),
	)

	add(
		name = '1505-5-P3D-BaseUpcactBc8',
		discrepancy = dict(
			net = dict(
				arch_core = 'v3',
			),
			train = dict(
				class_weights = [1, 8],
				**dset_P3D,
			),
		),
	)

	add(
		name = '1505-6-P3D-SumUpcactBc8',
		discrepancy = dict(
			net = dict(
				arch_core = 'v3',
				arch_classifier = 'sumSimple',
			),
			train = dict(
				class_weights = [1, 8],
				**dset_P3D,
			),
		),
	)
	


	add(
		name = '1505-7-P3D-BaseFoc8',
		discrepancy = dict(
			train = dict(
				class_weights = [1, 8],
				loss_name = "focal",
				focal_loss_gamma = 3.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
				**dset_P3D,
			),
		),
	)

	add(
		name = '1505-8-P3D-pwFfModpsmBc8',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'perswaveFlatfixedModpsm', 'persDirect'],
				arch_feature_procs_sticky = ['perswaveFlatfixedModpsm', 'persDirect'],
			),
			train = dict(
				class_weights = [1, 8],
				**dset_P3D,
			)
		),
	)

	add(
		name = '1505-9-P3D-CrSumNeighbourD2StckBc8',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'neighboursD2', 'persDirect'],
				arch_feature_procs_sticky = ['neighboursD2', 'persDirect'],
				arch_classifier = 'sumSimple',
			),
			train = dict(
				class_weights = [1, 8],
				**dset_P3D,
			)
		),
	)
	

	add(
		name = '1506-1-P3D-Soup',
		discrepancy = dict(
			net = dict(
				arch_core = 'soup1',
				arch_feature_procs = ['pass'],
			),
			train = dict(
				**dset_P3D,
			)
		),
	)

	add(
		name = '1506-2-P3D-SoupFoc',
		discrepancy = dict(
			net = dict(
				arch_core = 'soup1',
				arch_feature_procs = ['pass'],
			),
			train = dict(
				loss_name = "focal",
				focal_loss_gamma = 2.0, # curve param
				focal_loss_alpha = 0.5, # scale of loss
				**dset_P3D,
			)
		),
	)

	add(
		name = '1507-1-P3D-S2Base',
		discrepancy = dict(
			net = dict(
				arch_classifier = 'sumAct2',
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		name = '1507-2-P3D-S2_NeighbourD2PdirectStck',
		discrepancy = dict(
			net = dict(
				arch_classifier = 'sumAct2',
				perspective = True,
				arch_feature_procs = ['pass', 'neighboursD2', 'persDirect'],
				arch_feature_procs_sticky = ['neighboursD2', 'persDirect'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		name = '1507-3-P3D-S2_PdirectStck',
		discrepancy = dict(
			net = dict(
				arch_classifier = 'sumAct2',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		name = '1507-4-P3D-S2_PencoderStck',
		discrepancy = dict(
			net = dict(
				arch_classifier = 'sumAct2',
				perspective = True,
				arch_feature_procs = ['pass', 'persEncoder'],
				arch_feature_procs_sticky = ['persEncoder'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		name = '1508-1-P3D-BaSpl-Base',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		name = '1508-2-P3D-BaSpl-PdirectStck',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',

				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	repeat('1508-1-P3D-BaSpl-Base', 2)
	repeat('1508-2-P3D-BaSpl-PdirectStck', 2)

	add(
		name = '1508-3-D2b-Base-PdirectStck',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				epoch_limit = 50,
			),
		),
	)

	cortest_base = extend_config(DEFAULTS, dict(
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	))
	
	cortest_1 = {
		'1509-1-P3D-BaSpl-PdNeighbourD2Bn': 'neighboursD2Bn',
		'1509-2-P3D-BaSpl-PdNeighbourD2BnEnc': 'neighboursD2BnEnc',
		'1509-3-P3D-BaSpl-PdNeighbourLcor': 'neighboursLcor',
		'1509-4-P3D-BaSpl-PdNeighbourLcorBox5': 'neighboursLcorBox5',
	}

	for expname, featname in cortest_1.items():
		add(base = cortest_base,
			name = expname,
			discrepancy = dict(
				net = dict(
					arch_feature_procs = ['pass', 'persDirect', featname],
				)
			)
		)

	exps_to_prolong = list(cortest_1.keys()) + [
		'1508-2-P3D-BaSpl-PdirectStck',
	]

	for n in exps_to_prolong:
		add(
			base = cfgs_by_name[n], 
			name = f'{n}-long80',
			discrepancy = dict(
				train = dict(
					epoch_limit = 80,
				),
			),
		)

	
	add(base = cortest_base,
		name = '1511-1-P3C-BaSpl-PdNeighbourLcor',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'persDirect', 'neighboursLcor'],
			),
			train = dict(
				**dset_P3C,
			),
		)
	)

	add(
		name = '1511-2-P3D-Soup-Pdirect',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_core = 'soup1',
				arch_feature_procs = ['pass', 'persDirect'],
			),
			train = dict(
				**dset_P3D,
			)
		),
	)

	add(
		name = '1511-3-P3D-Soup-PdNeighbourLcor',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_core = 'soup1',
				arch_feature_procs = ['pass', 'persDirect', 'neighboursLcor'],
			),
			train = dict(
				**dset_P3D,
			)
		),
	)

	add(
		name = '1511-4-P3D-Soup-PdNeighbourD2BnEnc',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_core = 'soup1',
				arch_feature_procs = ['pass', 'persDirect', 'neighboursD2BnEnc'],
			),
			train = dict(
				**dset_P3D,
			)
		),
	)

	add(base = cortest_base,
		name = '1511-5-P3D-BaSpl-PdPoolcor',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'persDirect', 'poolcor'],
			),
			train = dict(
				**dset_P3D,
			),
		)
	)

	add(base = cortest_base,
		name = '1511-6-P3D-BaSpl-PdPoolcorPrc',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'persDirect', 'poolcorPrc'],
			),
			train = dict(
				**dset_P3D,
			),
		)
	)

	add(base = cortest_base,
		name = '1511-7-P3D-BaSpl-PdPoolcorPrcCat',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'persDirect', 'poolcorPrcCat'],
			),
			train = dict(
				**dset_P3D,
			),
		)
	)

	add(base = cortest_base,
		name = '1511-8-P3D-BaSpl-PdNeighbourD2BnEncStck',
		discrepancy = dict(
			net = dict(
				arch_feature_procs = ['pass', 'persDirect', 'neighboursD2BnEnc'],
				arch_feature_procs_sticky = ['persDirect', 'neighboursD2BnEnc'],
			),
			train = dict(
				**dset_P3D,
			),
		)
	)

	add(base = cortest_base,
		name = '1512-1-P3D-BaSpl-NeighbourD2BnEncStck',
		discrepancy = dict(
			net = dict(
				perspective = False,
				arch_feature_procs = ['pass', 'neighboursD2BnEnc'],
				arch_feature_procs_sticky = ['neighboursD2BnEnc'],
			),
			train = dict(
				**dset_P3D,
			),
		)
	)

	add(base = cortest_base,
		name = '1512-2-P3D-BaSpl-PdPdeNeighbourD2BnEncStck',
		discrepancy = dict(
			net = dict(
				perspective = True,
				arch_feature_procs = ['pass', 'neighboursD2BnEnc', 'persDirect', 'persEncoder'],
				arch_feature_procs_sticky = ['neighboursD2BnEnc', 'persDirect', 'persEncoder'],
			),
			train = dict(
				**dset_P3D,
			),
		)
	)


	AUG_NOISE_PRESETS = [dict(
		layers = (
			(0.18, 1.0),
			(0.31, 0.5),
			(0.84, 0.2),
		),
		magnitude_range = [0.1, 0.6],
	), dict(
		layers = (
			(0.031, 1.0),
			(0.451, 0.3),
		),
		magnitude_range = [0.2, 0.8],
	), dict(
		layers = (
			(1/256, 1.0),
			(0.18, 0.1),
		),
		magnitude_range = [0.1, 2],
	), dict(
		layers = (
			(0.58512, 1.0),
			(0.03412, 0.3),
		),
		magnitude_range = [0.1, 2.2],
	), dict(
		layers = (
			(0.7141, 1.0),
			(0.0412, 0.2),
		),
		magnitude_range = [0.2, 2],
	)]

	NOISE4 = AUG_NOISE_PRESETS[3]

	for i, ndef in enumerate(AUG_NOISE_PRESETS):
		ni = i+1
		add(
			name = f'1513-{ni}-P3D-Noise{ni}-BaSpl',
			discrepancy = dict(
				net = dict(
					arch_core = 'v4split',
					arch_classifier = 'sumSimple',
					perspective = True,
					arch_feature_procs = ['pass'],
					arch_feature_procs_sticky = [],
				),
				train = dict(
					augmentation_noise = ndef,
					**dset_P3D,
				),
			),
		)
		add(
			name = f'1513-{ni}-P3D-Noise{ni}-BaSpl-PdirectStck',
			discrepancy = dict(
				net = dict(
					arch_core = 'v4split',
					arch_classifier = 'sumSimple',
					perspective = True,
					arch_feature_procs = ['pass', 'persDirect'],
					arch_feature_procs_sticky = ['persDirect'],
				),
				train = dict(
					augmentation_noise = ndef,
					**dset_P3D,
				),
			),
		)

	repeat(f'1513-4-P3D-Noise4-BaSpl', 4)
	repeat(f'1513-4-P3D-Noise4-BaSpl-PdirectStck', 4)
	repeat(f'1513-5-P3D-Noise5-BaSpl-PdirectStck', 2)


	add(
		base = cfgs_by_name['1508-1-P3D-BaSpl-Base'],
		name = '1513-4-P3D-Noise1-BaSpl-Base',
		discrepancy = dict(
			train = dict(
				augmentation_noise = AUG_NOISE_PRESETS[0],
				#**FOC1,
			),
		),
	)

	FOC1 = dict(
		loss_name = "focal",
		focal_loss_gamma = 2.0, # curve param
		focal_loss_alpha = 0.5, # scale of loss
	)

	add(
		base = cfgs_by_name['1508-1-P3D-BaSpl-Base'],
		name = '1515-1-P3D-BaSpl-BaseFoc',
		discrepancy = dict(
			train = dict(
				**FOC1,
			),
		),
	)

	add(
		base = cfgs_by_name['1508-2-P3D-BaSpl-PdirectStck'],
		name = '1515-2-P3D-BaSpl-PdirectStckFoc',
		discrepancy = dict(
			train = dict(
				**FOC1,
			),
		),
	)

	repeat('1515-2-P3D-BaSpl-PdirectStckFoc', 1)
	
	add(
		base = cfgs_by_name['1508-2-P3D-BaSpl-PdirectStck'],
		name = '1515-3-P3D-Noise1-BaSpl-PdirectStckFoc',
		discrepancy = dict(
			train = dict(
				augmentation_noise = AUG_NOISE_PRESETS[0],
				**FOC1,
			),
		),
	)

	add(
		base = cfgs_by_name['1511-2-P3D-Soup-Pdirect'],
		name = '1515-4-P3D-Soup-PdirectFoc',
		discrepancy = dict(
			train = dict(
				**FOC1,
			),
		),
	)

	add(
		name = '1516-1-P3D-BaSplSkip1-Base',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				arch_backbone = 'resnext101_32x8d-skip1',
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	repeat('1516-1-P3D-BaSplSkip1-Base', 2)

	add(
		name = '1516-2-P3D-BaSplSkip2-Base',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				arch_backbone = 'resnext101_32x8d-skip2',
			),
			train = dict(
				**dset_P3D,
			),
		),
	)


	add(
		name = '1516-3-P3D-BaSplSkip1-PdirectStck',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				arch_backbone = 'resnext101_32x8d-skip1',

				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	repeat('1516-3-P3D-BaSplSkip1-PdirectStck', 2)

	add(
		base = cfgs_by_name['1511-2-P3D-Soup-Pdirect'],
		name = '1516-4-P3D-SoupSkip1-PdirectFoc',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d-skip1',
			),
			train = dict(
				**FOC1,
			),
		),
	)

	add(
		name = '1516-5-P3D-BaSplSkip2-PdirectStck',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				arch_backbone = 'resnext101_32x8d-skip2',

				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		name = '1516-6-P3D-BaSplSkip1-PdNeighbourD2BnEnc',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				arch_backbone = 'resnext101_32x8d-skip1',

				perspective = True,
				arch_feature_procs = ['pass', 'persDirect', 'neighboursD2BnEnc'],
				arch_feature_procs_sticky = ['persDirect', 'neighboursD2BnEnc'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)
	add(
		name = '1516-7-P3D-BaSplSkip2-PdNeighbourD2BnEnc',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				arch_backbone = 'resnext101_32x8d-skip2',

				perspective = True,
				arch_feature_procs = ['pass', 'persDirect', 'neighboursD2BnEnc'],
				arch_feature_procs_sticky = ['persDirect', 'neighboursD2BnEnc'],
			),
			train = dict(
				**dset_P3D,
			),
		),
	)

	add(
		base = cfgs_by_name['1509-2-P3D-BaSpl-PdNeighbourD2BnEnc'],
		name = '1517-1-P3D-Noise4-BaSpl-PdNeighbourD2BnEnc',
		discrepancy = dict(
			train = dict(
				augmentation_noise = NOISE4,
			),
		),
	)

	add(
		base = cfgs_by_name['1509-3-P3D-BaSpl-PdNeighbourLcor'],
		name = '1517-2-P3D-Noise4-BaSpl-PdNeighbourLcor',
		discrepancy = dict(
			train = dict(
				augmentation_noise = NOISE4,
			),
		),
	)
	
	# variant without fusion
	add(
		name = '1518-1-P3D-Noise4-BaSpl-BaseNoSum',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)
	repeat('1518-1-P3D-Noise4-BaSpl-BaseNoSum', 5)


	# synth variant without perspective aware object sizes
	add(
		name = '1518-2-D2b-Noise4-BaSpl-PdirectStck',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
				epoch_limit = 50,
			),
		),
	)
	repeat('1518-2-D2b-Noise4-BaSpl-PdirectStck', 5)

	# synth variant without perspective aware object sizes
	add(
		name = '1518-3-P3D-Noise4-BaSpl-PdirectEncStck',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'sumSimple',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect', 'persEncoder'],
				arch_feature_procs_sticky = ['persDirect', 'persEncoder'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)
	repeat('1518-3-P3D-Noise4-BaSpl-PdirectEncStck', 3)

	# no fusion but with perspective
	add(
		name = '1518-4-P3D-Noise4-BaSpl-NoSumPdirect',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)
	
	add(
		name = '1525-3-P3A-Noise4-BaSpl-NoSumPdirect',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3A,
			),
		),
	)

	add(
		name = '1525-4-P3E-Noise4-BaSpl-NoSumPdirect',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3E,
			),
		),
	)

	add(
		name = '1526-1-P3Dbs-Noise4-BaSpl-NoSumPdirect',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
					dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp3Dbs_cityscapes-train',
					dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp3Dbs_cityscapes-val',
					epoch_limit = 50,
			),
		),
	)

	add(
		name = '1526-2-P3Dsc-Noise4-BaSpl-NoSumPdirect',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3p4sc_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3p4sc_cityscapes-val',
				epoch_limit = 50,
			),
		),
	)

	for dsetvar in ['v3Dsz02', 'v3Dsz07', 'v3Dsz05', 'v3Dsz10', 'v3Dsz6w']:
		cf = dict(
			name = f'1526-4x-{dsetvar}-Noise4-BaSpl-NoSumPdirect',
			discrepancy = dict(
				net = dict(
					arch_core = 'v4split',
					arch_classifier = 'last',
					perspective = True,
					arch_feature_procs = ['pass', 'persDirect'],
					arch_feature_procs_sticky = ['persDirect'],
				),
				train = dict(
					augmentation_noise = NOISE4,
					dset_train = f'1230_SynthObstacle_Fusion_Fblur5-{dsetvar}_cityscapes-train',
					dset_val = f'1230_SynthObstacle_Fusion_Fblur5-{dsetvar}_cityscapes-val',
					epoch_limit = 50,
				),
			),
		)
		add(**cf)

		c = extend_config(cf, dict(
			name = f'15-2010-01-{dsetvar}-setr1',
			discrepancy = dict(
				net = dict(
					arch_feature_procs = ['pass', 'attentropy_SETR1'],
					arch_feature_procs_sticky = [],
				),
				train = dict(
					mod_sampler = 'v1-768-entSETR',
					num_workers = 12,
				),
				extra_features = {
					'attentropy': 'SETR',
				},
			),
		))
		add(**c)

		c = extend_config(cf, dict(
			name = f'15-2010-02-{dsetvar}-NoPers',
			discrepancy = dict(
				net = dict(
					arch_feature_procs = ['pass'],
					arch_feature_procs_sticky = [],
				),
			),
		))
		add(**c)


	add(
		name = '1526-3-P3D-Noise4-BaSpl-ZonesFixed',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'zoneFixed8p'],
				arch_feature_procs_sticky = ['zoneFixed8p'],

			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D
			),
		),
	)

	add(
		name = '1526-3-D2b-Noise4-BaSpl-NoPers',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [],
			),
			train = dict(
				augmentation_noise = NOISE4,
				dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
				epoch_limit = 50,
			),
		),
	)

	add(
		name = '1526-5-P3D-Noise4-BaSpl-YXdirect',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'YXdirect'],
				arch_feature_procs_sticky = ['YXdirect'],

			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D
			),
		),
	)

	add(
		name = '1526-5-D2b-Noise4-BaSpl-YXdirect',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'YXdirect'],
				arch_feature_procs_sticky = ['YXdirect'],

			),
			train = dict(
				augmentation_noise = NOISE4,
				dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
				epoch_limit = 50,
			),
		),
	)


	add(
		name = '1527-1-P3D-Noise4-Dil025-NoSumPdirect',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4splitDil025',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1527-2-P3D-Noise4-Dil025-ZonesFixed',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4splitDil025',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'zoneFixed8p'],
				arch_feature_procs_sticky = ['zoneFixed8p'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1527-3-P3D-Noise4-PolyLR-ZonesFixed',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'zoneFixed8p'],
				arch_feature_procs_sticky = ['zoneFixed8p'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				optimizer = dict(
					opt_type = 'poly',
				),
				**dset_P3D,
			),
		),
	)

	add(
		name = '1527-4-P3D-Noise4-WeightDecay-ZonesFixed',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'zoneFixed8p'],
				arch_feature_procs_sticky = ['zoneFixed8p'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				optimizer = dict(
					weight_decay = 5e-4,
				),
				**dset_P3D,
			),
		),
	)

	add(
		name = '1527-7-P3D-HanetOrigBaseWd',
		discrepancy = dict(
			net = dict(
				arch_core = 'hanetOrigBase',
				perspective = True,
				batch_train = 6,
			),
			train = dict(
				augmentation_noise = NOISE4,
				optimizer = dict(
					weight_decay = 5e-4,
					opt_type = 'poly',
				),
				**dset_P3D,
			),
		),
	)

	add(
		name = '1527-5-P3D-HanetOrigYXWd',
		discrepancy = dict(
			net = dict(
				arch_core = 'hanetOrigYX',
				perspective = True,
				batch_train = 6,
			),
			train = dict(
				augmentation_noise = NOISE4,
				optimizer = dict(
					weight_decay = 5e-4,
					opt_type = 'poly',
				),
				**dset_P3D,
			),
		),
	)

	add(
		name = '1527-6-P3D-HanetOrigPXWd',
		discrepancy = dict(
			net = dict(
				arch_core = 'hanetOrigPX',
				perspective = True,
				batch_train = 6,
			),
			train = dict(
				augmentation_noise = NOISE4,
				optimizer = dict(
					weight_decay = 5e-4,
					opt_type = 'poly',
				),
				**dset_P3D,
			),
		),
	)

	add(
		name = '1518-5-D2b-Noise4-BaSpl-NoSumPdirect',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				dset_train = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5-v2b_cityscapes-val',
				epoch_limit = 50,
			),
		),
	)

	add(
		name = '1518-6-P3D-Noise4-BaSpl-NoSumPdirectFuser1',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'fuserSimple',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = ['persDirect'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)
	repeat('1518-6-P3D-Noise4-BaSpl-NoSumPdirectFuser1', 5)


	add(
		name = '1519-1-P3D-Noise4-BaSpl-NoSumPdirectNoStck',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect'],
				arch_feature_procs_sticky = [''],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1519-2-P3D-Noise4-BaSpl-NoSumPdirectPers1Only',
		discrepancy = dict(
			net = dict(
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirectLayer1'],
				arch_feature_procs_sticky = [''],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1519-3-P3D-Noise4-BaSpl-PmapBgbone4Ch',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d-pmap4Ch',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [''],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1525-1-D2b-Noise4-BaSpl-PmapBgbone4Ch',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d-pmap4Ch',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [''],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_D2b,
			),
		),
	)

	add(
		name = '1519-4-P3D-Noise4-BaSpl-PmapBgboneBranch',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d-pmapBranch',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [''],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1525-2-D2b-Noise4-BaSpl-PmapBgboneBranch',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d-pmapBranch',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [''],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_D2b,
			),
		),
	)

	add(
		name = '1519-5-P3D-Noise4-BaSpl-Unwarp',
		unwarp = True,
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = False,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [''],
			),
			train = dict(
				augmentation_noise = NOISE4,
				dset_train = '1230_SynthObstacle_Fusion_Fblur5unwarp1-v3persp3D_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5unwarp1-v3persp3D_cityscapes-val',
				epoch_limit = 50,
			),
		),
	)

	add(
		name = '1520-1-P3D-Noise4-BaSpl-ZonesFixed',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'zoneFixed8p'],
				arch_feature_procs_sticky = ['zoneFixed8p'],

			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D
			),
		),
	)

	add(
		name = '1520-2-P3D-Noise4-BaSpl-ZonesTrain',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'zoneTrain8p'],
				arch_feature_procs_sticky = ['zoneTrain8p'],

			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D
			),
		),
	)

	add(
		name = '1520-3-P3D-Noise4-BaSpl-Enc',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				arch_feature_procs = ['pass', 'persDirect', 'persEncoder'],
				arch_feature_procs_sticky = ['persDirect', 'persEncoder'],

			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D
			),
		),
	)



	add(
		name = '1601-N4-Base',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = False,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1601-N4-Opt1',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = False,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [],
			),
			train = dict(
				augmentation_noise = NOISE4,
				dset_train = '1230_SynthObstacle_Fusion_FOpt1-v3persp3D_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_FOpt1-v3persp3D_cityscapes-val',
				epoch_limit = 50,
			),
		),
	)

	add(
		name = '1560-1-P3D-WeightedSoup1x1Sig',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'wsoup-1x1sig',
				perspective = True,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)
	add(
		name = '1560-2-P3D-WeightedSoupZoneSig',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'wsoup-zonesig',
				perspective = True,
				arch_feature_procs = ['pass'],
				arch_feature_procs_sticky = [],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)

	add(
		name = '1580-1-P3D-Normals1',
		discrepancy = dict(
			net = dict(
				arch_backbone = 'resnext101_32x8d',
				arch_core = 'v4split',
				arch_classifier = 'last',
				perspective = True,
				normals = True,
				arch_feature_procs = ['pass', 'persDirect', 'normals1'],
				arch_feature_procs_sticky = ['persDirect', 'normals1'],
			),
			train = dict(
				augmentation_noise = NOISE4,
				**dset_P3D,
			),
		),
	)



	return cfgs


def exp15_configs_unwarp():
	
	DEFAULTS = dict(
		discrepancy = dict(
			net = dict(
				batch_train = 12,
				batch_eval = 6,
				batch_infer = 1,

				separate_gen_image = False,
				perspective = False,

				arch_core = 'v1',
				arch_backbone = 'resnext101_32x8d',
				arch_feature_procs = ['pass'],
				arch_classifier = 'last',
			),
			train = dict(
				dset_train = '1230_SynthObstacle_Fusion_Fblur5-v3persp3D_cityscapes-train',
				dset_val = '1230_SynthObstacle_Fusion_Fblur5-v3persp3D_cityscapes-val',
				mod_sampler = 'v1-768',

				num_workers = 4,
				epoch_limit = 50,

				# loss
				loss_name = 'cross_entropy',
				class_weights = None,
				augmentation_noise = dict(
					layers = (
						(0.58512, 1.0),
						(0.03412, 0.3),
					),
					magnitude_range = [0.1, 2.2],
				),
			),
			preproc_blur = False,
		)
	)

	cfgs = []
	cfgs_by_name = {}

	def add(base=DEFAULTS, REPEATS=4, **diff):
		c = extend_config(base, diff)
		basename = c['name']
		cfgs.append(c)
		cfgs_by_name[basename] = c

		if REPEATS and REPEATS > 0:
			for i in range(REPEATS):
				add(base=cfgs_by_name[basename], REPEATS=0, name=f'{basename}-rep{i+1}')

		return c

	return cfgs


if __name__ == '__main__':
	cfgs = exp15_configs()
	names = [cfg['name'] for cfg in cfgs]
	print(','.join(names))

