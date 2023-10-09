

from .jupyter_show_image import show, adapt_img_data, imread, imwrite

# import numpy as np

# from io import BytesIO
# from PIL import Image as PIL_Image

# try:
# 	from ipywidgets import HBox, VBox, Box, Image
# 	HAS_IPYWIDGETS = True
# except:
# 	HAS_IPYWIDGETS = False

# from IPython.display import display, Image as ipy_Image
# from matplotlib import cm

# # fix IPython.display.Image to embed webp
# ipy_Image._ACCEPTABLE_EMBEDDINGS.append('webp')
# ipy_Image._MIMETYPES['webp'] = 'image/webp'


# def imread(path):
# 	return np.asarray(PIL_Image.open(path))

# IMWRITE_OPTS = dict(
# 	webp = dict(quality = 85),
# )

# def imwrite(path, data, format=None):
# 	path = Path(path)
# 	path.parent.mkdir(exist_ok=True, parents=True)
	
# 	# log.info(f'write {path}')

# 	try:
# 		PIL_Image.fromarray(data).save(
# 			path, 
# 			format = format,
# 			**IMWRITE_OPTS.get(path.suffix.lower()[1:], {}),
# 		)
# 	except Exception as e:
# 		log.exception(f'Saving {path}')


# def jupyter_img_from_data(img_data, compression='webp'):
# 	with BytesIO() as buffer:
# 		PIL_Image.fromarray(img_data).save(
# 			buffer, 
# 			format = format,
# 		)
# 		#imwrite(buffer, img_data, format=compression)
# 		# 		jup_im = Image(data=buffer.getvalue(), format=compression)
# 		jup_im = Image(value=buffer.getvalue(), format=compression)

# 	return jup_im


# def display_image_jpg(img_data):
# 	im = jupyter_img_from_data(img_data)
# 	display(im)
# 	return img_data

# def adapt_img_data(img_data, cmap_pos=cm.get_cmap('magma'), cmap_div=cm.get_cmap('Spectral').reversed(), ret_compr=False):
# 	num_dims = img_data.shape.__len__()
# 	c = 'jpg'

# 	if num_dims == 3:
# 		if img_data.shape[2] > 3:
# 			img_data = img_data[:, :, :3]

# 		if img_data.dtype != np.uint8 and np.max(img_data) < 1.1:
# 			img_data = (img_data * 255).astype(np.uint8)


# 	elif num_dims == 2:
# 		if img_data.dtype == bool:

# 			img_data = img_data.astype(np.uint8)*255
# 			c = 'png'

# 		else:
# 			vmax = np.max(img_data)
# 			if img_data.dtype == np.uint8 and vmax == 1:
# 				img_data = img_data * 255

# 			else:
# 				vmin = np.min(img_data)

# 				if vmin >= 0:
# 					img_data = (img_data - vmin) * (1 / (vmax - vmin))
# 					img_data = cmap_pos(img_data, bytes=True)[:, :, :3]

# 				else:
# 					vrange = max(-vmin, vmax)
# 					img_data = img_data / (2 * vrange) + 0.5
# 					img_data = cmap_div(img_data, bytes=True)[:, :, :3]

# 	if ret_compr:
# 		return img_data, c
# 	else:
# 		return img_data


# def jup_widget_from_img_data(img_data, cmap_pos=cm.get_cmap('magma'), cmap_div=cm.get_cmap('Spectral').reversed()):
# 	img_data, c = adapt_img_data(img_data, cmap_pos = cmap_pos, cmap_div = cmap_div, ret_compr=True)

# 	return jupyter_img_from_data(img_data, compression=c)


# def jup_widget_from_col(cols):
# 	if not isinstance(cols, (list, tuple)):
# 		cols = [cols]

# 	jup_ws = [jup_widget_from_img_data(img_data) for img_data in cols]

# 	# 	return HBox(jup_ws) if jup_ws.__len__() > 1 else jup_ws[0]

# 	# 	size_per_elem = 95 / jup_ws.__len__()
# 	# 	size_css = '{s:02d}%'.format(s=np.floor(size_per_elem).astype(np.int))

# 	# 	for e in jup_ws:
# 	# 		e.layout.width = size_css
# 	# 		e.layout.margin = '1%'

# 	wrapped = []
# 	for im in jup_ws:
# 		im.layout.width = '98%'

# 		w = Box([im])
# 		w.layout.display = 'block'
# 		w.layout.overflow = 'hidden'
# 		w.layout.margin = '0.5%'

# 		wrapped.append(w)

# 	jup_ws = wrapped

# 	if jup_ws.__len__() > 1:

# 		box = HBox(jup_ws)
# 		# 		box.layout.display = 'flex'

# 		s = box.layout

# 		# 		s.width = '100%'
# 		# 		s.display = 'block'

# 		s.display = 'flex'
# 		s.flex_flow = 'row nowrap'
# 		s.overflow = 'hidden'
# 		# s.overflow_x = 'hidden'

# 		s.justify_content = 'space-around'
# 		s.align_content = 'center'

# 		return box
# 	else:
# 		jup_ws[0].layout.margin = '0.5%'
# 		return jup_ws[0]


# def jup_widget_from_rows(rows):
# 	jup_ws = [jup_widget_from_col(row) for row in rows]
# 	return VBox(jup_ws) if jup_ws.__len__() > 1 else jup_ws[0]


# def show(*rows):
# 	jup_widget = jup_widget_from_rows(rows)
# 	display(jup_widget)

# if not HAS_IPYWIDGETS:
# 	def show(img_data, compression = None):

# 		img_data, compression_suggested = adapt_img_data(img_data, ret_compr=True)

# 		compression = compression or compression_suggested

# 		with BytesIO() as buffer:
# 			imageio.imwrite(buffer, img_data, compression)
# 			jup_im = ipy_Image(data=buffer.getvalue(), format=compression, embed=True)
			
# 		display(jup_im)



# # https://stackoverflow.com/questions/21103622/auto-resize-image-in-css-flexbox-layout-and-keeping-aspect-ratio