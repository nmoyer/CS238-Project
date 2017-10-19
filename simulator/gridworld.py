import Tkinter as tk
import time
import random
from numpy.random import normal

class Tile:

	def __init__(self, ground_truth):
		self.ground_truth = ground_truth
		self.prob_safe = 0

	def getGroundTruth(self):
		return self.ground_truth

	def setGroundTruth(self, ground_truth):
		self.ground_truth = ground_truth

	def getProbSafe(self):
		return self.prob_safe

	def setProbSafe(self, prob_safe):
		self.prob_safe = prob_safe

	def getColor(self):
		if self.prob_safe == 0:
			return 'black'
		elif self.prob_safe == 1:
			return 'white'

		gray_num = int(self.prob_safe*100)
		return 'gray' + str(gray_num)

class Grid:

	def __init__(self, height, width, num_clusters, cluster_dim):
		self.height = height
		self.width = width
		self.num_clusters = num_clusters
		self.cluster_dim = cluster_dim

		self.grid = [[Tile(1) for _ in range(width)] for _ in range(height)]

		def makeCluster(grid, cluster_center, cluster_dim):
			gradient = 1.0/cluster_dim
			for radius in range(cluster_dim,0,-1):
				if radius == 0: 
					grid[cluster_center[0]][cluster_center[1]].setGroundTruth(0)
					continue

				for row in range(-radius, radius):
					for col in range(-radius, radius):
						tile = grid[cluster_center[0] + row][cluster_center[1] + col]
						tile.setGroundTruth(gradient*radius)

		for _ in range(self.num_clusters):
			row_start = random.randint(self.cluster_dim,self.height-self.cluster_dim)
			col_start = random.randint(self.cluster_dim,self.width-self.cluster_dim)
			makeCluster(self.grid, (row_start, col_start), self.cluster_dim)

	def getHeight(self):
		return self.height

	def getWidth(self):
		return self.width

	def getTileColor(self, row, col):
		return self.grid[row][col].getColor()

	def update(self, tiles, confidence):
		for tile_indices in tiles:
			tile = self.grid[tile_indices[0]][tile_indices[1]]
			observed_prob = min(1,max(0,normal(tile.getGroundTruth(), confidence)))
			tile.setProbSafe(observed_prob)

class GridWorld:

	def __init__(self, iteration_rate, grid, aircraft, tk_root):
		self.grid = grid
		self.current_time = 0
		self.iteration_rate = iteration_rate
		self.aircraft = aircraft

		self.canvas_height = 10*self.grid.getHeight()
		self.canvas_width = 10*self.grid.getWidth()
		self.canvas = tk.Canvas(tk_root, width=self.canvas_width, height=self.canvas_height, background='black')
		self.canvas.grid(row=0,column=0)

		self.canvas_grid = [[self.canvas.create_rectangle(col, row, col+10, row+10, fill='black') 
							for col in range(0, self.canvas_width, 10)]
							for row in range(0, self.canvas_height, 10)]

	def update(self):
		aircraft_move, aircraft_sensor = self.aircraft.getAction()
		self.aircraft.updateLocation(aircraft_move)
		if aircraft_sensor:
			self.aircraft.useSensor(aircraft_sensor)

		if self.aircraft.isCrash():
			return -1 
		return 1

	def render(self, tk_root):
		for row in range(self.canvas_height/10): 
			for col in range(self.canvas_width/10):
				self.canvas.itemconfig(self.canvas_grid[row][col], fill=self.grid.getTileColor(row, col))

		aircraft_row, aircraft_col = self.aircraft.getLocation()
		self.canvas.itemconfig(self.canvas_grid[aircraft_row][aircraft_col], fill='blue')
		
		tk_root.update()
		time.sleep(self.iteration_rate)


