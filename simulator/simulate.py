from Tkinter import *
from gridworld import *
from agent import *

grid = Grid(50, 50, 10, 5)
energy_source = EnergySource(100)
sensor = Sensor((1,1), ((0,0), 30, 30), 0.1)
nom_traject = []
aircraft = Aircraft(energy_source, [sensor], 1, nom_traject, grid)

tk_root = Tk()
gridworld = GridWorld(0.2, grid, aircraft, tk_root)

num_iterations = 100
for iteration in range(num_iterations):
	if gridworld.update() < 0:
		break
	gridworld.render(tk_root)

tk_root.mainloop()