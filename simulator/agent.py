class EnergySource:

	def __init__(self, max_battery):
		self.max_battery = max_battery
		self.current_battery = max_battery

	def getCurrentBattery(self):
		return self.getCurrentBattery

	def setCurrentBattery(self, level):
		self.current_battery = level

	def useBattery(self, usage_rate, usage_time):
		self.current_battery -= usage_rate*usage_time

class Sensor:

	def __init__(self, energy_usage, coverage, confidence):
		self.energy_rate = energy_usage[0]
		self.energy_time = energy_usage[1]
		self.coverage = coverage
		self.confidence = confidence

	def getEnergyRate(self):
		return self.energy_rate

	def setEnergyRate(self, energy_rate):
		self.energy_rate = energy_rate

	def getEnergyTime(self):
		return self.energy_time

	def setEnergyTime(self, energy_time):
		self.energy_time = energy_time

	def getCoverage(self):
		return self.coverage

	def getConfidence(self):
		return self.confidence

class Aircraft:

	def __init__(self, energy_source, sensors, speed, nom_traject, grid):
		self.energy_source = energy_source
		self.sensors = sensors
		self.location = (0,0)
		self.speed = speed
		self.nom_traject = nom_traject
		self.grid = grid

	def getLocation(self):
		return self.location

	def setLocation(self, location):
		self.location = location

	def getSpeed(self):
		return self.speed

	def setSpeed(self, speed):
		self.speed = speed

	def updateLocation(self, move):
		curr_location = self.location
		self.location = (curr_location[0] + self.speed*move[0], 
						 curr_location[1] + self.speed*move[1])

	def getAction(self):
		move = (1,1)
		sensor = self.sensors[0]
		return (move,sensor)

	def isCrash(self):
		if self.location[0] >= self.grid.getHeight() or self.location[0] < 0 or\
		   self.location[1] >= self.grid.getWidth() or self.location[1] < 0:
			return True

		return False

	def useSensor(self, sensor):
		self.energy_source.useBattery(sensor.getEnergyRate(), sensor.getEnergyTime())
		self.illuminate(sensor)

	def illuminate(self, sensor):
		coverage_offset, coverage_height, coverage_width = sensor.getCoverage()
		start_row = self.location[0] + coverage_offset[0]
		start_col = self.location[1] + coverage_offset[1]

		tiles = []
		for row_offset in range(coverage_height):
			for col_offset in range(coverage_width):
				row = start_row+row_offset
				col = start_col+col_offset
				if row < 0 or row >= self.grid.getHeight() or\
				   col < 0 or col >= self.grid.getWidth():
					continue
				tiles.append((row, col))

		self.grid.update(tiles, sensor.getConfidence())



