import itertools
from copy import deepcopy

def circle(center, radius, grid_len):
	offsets = [[0,1],[1,0],[0,-1],[-1,0]]
	locations = []
	for d in range(1,radius+1):
		for combination in itertools.combinations_with_replacement(offsets, d):
			location = deepcopy(center)
			for direction in combination:
				location[0] += direction[0]
				location[1] += direction[1]
			# if location[0] > grid_len or location[0] <= 0 or \
			#    location[1] > grid_len or location[1] <= 0:
			#    continue
		 	locations.append(location+[d])
	return locations

print(circle([0,0],4,10))