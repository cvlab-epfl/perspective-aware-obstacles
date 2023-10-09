

* produce batch - *segmentation_reder.py*
```python
self.mean = [123.68, 116.779, 103.939]
self.std = [70.59564226, 68.52497082, 71.41913876]

...
batch['mean'] = self.mean
batch['std'] = self.std
img = transform.normalize(img, self.mean, self.std)
img = transform.numpy_to_torch_image(img)
batch['image'] = img
batch['mean'] = self.mean
batch['std'] = self.std
batch['name'] = self.names[idx]
if labels is not None:
	labels = torch.LongTensor(labels)
	batch['labels'] = labels
	batch['target_size'] = labels.shape[:2]
else:
	batch['target_size'] = img.shape[:2]
```

* run network and save outputs - *inference.py*
```python
for step, batch in enumerate(wd_data_loader):
	...
	pred, pred_w_outlier, conf_probs = evaluation.segment_image(model, batch, args, conf_mats, ood_id, num_classes)

	store_outputs(batch, pred, pred_w_outlier, conf_probs)
```

* interpret outputs - *inference.py*
```python
def store_outputs(batch, pred, pred_w_outlier, conf_probs):
	pred = pred.detach().cpu().numpy().astype(np.int32)
	pred_w_outlier = pred_w_outlier.detach().cpu().numpy().astype(np.int32)
	conf_probs = conf_probs.detach().cpu().numpy()
	img_raw = transform.denormalize(
		batch['image'][0],
		batch['mean'][0].numpy(), batch['std'][0].numpy(),
	)
	true = batch['labels'][0].numpy().astype(np.int32)
	name = batch['name'][0]
	store_images(img_raw, pred, true, class_info, 'segmentation/'+name)
	store_images(img_raw, pred_w_outlier, true, class_info, 'seg_with_conf/'+ name)
	store_conf(img_raw, conf_probs, name, 'probs')

def store_images(img_raw, pred, true, class_info, name):
	img_pred = colorize_labels(pred, class_info)

	error_mask = np.ones(img_raw.shape)
	if true is not None:
		img_true = colorize_labels(true, class_info)
		img_errors = img_pred.copy()
		correct_mask = pred == true
		error_mask = pred != true
		ignore_mask = true == ignore_id
		img_errors[correct_mask] = 0
		img_errors[ignore_mask] = 0
		num_errors = error_mask.sum()
		img1 = np.concatenate((img_raw, img_true), axis=1)
		img2 = np.concatenate((img_errors, img_pred), axis=1)
		img = np.concatenate((img1, img2),axis=0)
		filename = '%s_%07d.jpg' % (name, num_errors)
		save_path = join(save_dir, filename)
	else:
		line = np.zeros((5, pred.shape[1], 3)).astype(np.uint8)
		img = np.concatenate((img_raw, line, img_pred), axis=0)
		save_path = join(save_dir, '%s.jpg' % (name))

	img = pimg.fromarray(img)
	saver_pool.apply_async(img.save, [save_path])

def store_conf(img_raw, conf, name, conf_type='logit'):
    conf = conf[0]
    img_conf = get_conf_img(img_raw, conf, 'conf_' + conf_type)
    img = np.concatenate([img_raw, img_conf], axis=0)
    save_path = join(save_dir, 'confidence',
                     '%s_%s.jpg' % (name, conf_type))
    img = pimg.fromarray(img)
    saver_pool.apply_async(img.save, [save_path])
```
