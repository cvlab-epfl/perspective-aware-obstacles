
def name_list(name_list):
	if not name_list:
		return []
	else:
		return [
			name for name in name_list.split(',') if name
		]
