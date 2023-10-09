from ..pipeline.log import log

from typing import Callable
from functools import partial

class Registry:
	def __init__(self):
		self.INITIALIZERS = {}
		self.MODULES = {}

	def register(self, name : str, init_func : Callable, *opts_s, **opts_kw):
		if opts_s or opts_kw:
			init_func = partial(init_func, *opts_s, **opts_kw)

		if name in self.INITIALIZERS or name in self.MODULES:
			log.warn(f'Module {name} is already registered')
		
		self.INITIALIZERS[name] = init_func

	def register_concrete(self, name : str, dset_object):

		if name in self.MODULES:
			log.warn(f'Module {name} is already registered')
		
		self.MODULES[name] = dset_object

		return dset_object

	def list_available_dsets(self):
		names = set(self.INITIALIZERS.keys()).union(self.MODULES.keys())
		names = list(names)
		names.sort()
		return names

	def get(self, name : str, cache=True):
		obj = self.MODULES.get(name)
		
		if obj is None or not cache:
			init_func = self.INITIALIZERS.get(name)

			if init_func is None:
				dslist = '\n '.join(self.list_available_dsets())
				# KeyError can't display newlines https://stackoverflow.com/questions/46892261/new-line-on-error-message-in-keyerror-python-3-3
				raise ValueError(f'No dataset called {name} in registry, avaiable datasets:\n {dslist}')

			else:
				obj = init_func()
				if cache:
					self.register_concrete(name, obj)

		return obj

	def register_class(self, *args, **kwargs):
		def decorator(class_to_register):
			configs = getattr(class_to_register, 'configs', None)

			if configs is None:
				def_cfg = getattr(class_to_register, 'default_cfg', None)
				if def_cfg is None:
					raise NotImplementedError(f'Class {class_to_register} has neither c.configs nor c.default_cfg')
				configs = [def_cfg]

			# config generator function
			if isinstance(configs, Callable):
				configs = configs()

			for cfg in configs:
				self.register(cfg['name'], partial(class_to_register, cfg))	

			return class_to_register

		return decorator



class ModuleRegistry:
	""" decorator """

	#registry_key_to_class = dict()

	baseclass_to_registry = dict()

	@staticmethod
	def key_from_baseclass(base_class):
		return base_class if isinstance(base_class, str) else base_class.__name__

	@staticmethod
	def key_from_baseclass_and_name(base_class, name):
		base_name = base_class if isinstance(base_class, str) else base_class.__name__
		return (base_name, name)

	@classmethod
	def get(cls, base_class, name):
		reg = cls.baseclass_to_registry[cls.key_from_baseclass(base_class)]
		return reg.get(name)

		# return cls.registry_key_to_class[
		# 	cls.key_from_baseclass_and_name(base_class, name)
		# ]

	def __init__(self, base_class, *_, **__):
		""" decorator construction """
		# self.key = self.key_from_baseclass_and_name(base_class, name)
		self.key = self.key_from_baseclass(base_class)
		self.reg = self.baseclass_to_registry.setdefault(self.key, Registry())


	def __call__(self, cls_to_register):
		""" decorator call """
		# self.registry_key_to_class[self.key] = cls_to_register
		# return cls_to_register
		return self.reg.register_class()(cls_to_register)



class ModuleRegistry2:
	""" decorator """

	CATEGORIES = {}

	@classmethod
	def get_implementation(cls, category, name):
		cat_entry = cls.CATEGORIES.get(category)
		if cat_entry is None:
			cats = ', '.join(cls.CATEGORIES.keys())
			raise KeyError(f'ModuleRegistry: no category {category}, categories available: {cats}')
		
		cls_and_cfg = cat_entry.get(name)
		if cls_and_cfg is None:
			names = ', '.join(cat_entry.keys())
			raise KeyError(f'ModuleRegistry: category {category}, no entry {name}, names available: {names}')

		class_entry = cls_and_cfg['cls']
		cfg_entry = cls_and_cfg['config']

		obj = class_entry(cfg_entry)

		return obj

	@classmethod
	def list_entries(cls, category):
		return cls.CATEGORIES[category].keys()

	def __init__(self, category=None, config_field='configs'):
		""" decorator construction """

		self.category = category
		self.config_field = config_field


	def __call__(self, cls_to_register):
		""" decorator call """

		# name, predefined or class name
		category = self.category or cls_to_register.__name__

		# extract configs from class
		configs = getattr(cls_to_register, self.config_field, [])
		# it can be a class method!
		if callable(configs):
			configs = configs()

		if configs:
			config_dict = self.CATEGORIES.setdefault(category, {})

			for cfg in configs:
				name = cfg['name']
				# if name in config_dict:
				# 	log.warning(f'Category {category}: Overriding config {name}')
				
				config_dict[name] = dict(cls=cls_to_register, config=cfg)
			
		else:
			log.warning(f'ModuleRegistry2: Class {cls_to_register.__name__} has no configs to register')

		return cls_to_register


