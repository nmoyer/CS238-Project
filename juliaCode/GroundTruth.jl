using Numbers
using PyCall
using Tasks

@pyimport Tkinter as tk

struct SimulatorState
	animation_rate::Float64
	canvas::PyObject
	canvas_grid::Array{PyObject,2}
end

function make_cluster(world_map::Array{Bool,2}, rng::AbstractRNG)
    cluster_height = 2
    cluster_width = 2

    row_start = Numbers.rand(rng, 1:(size(world_map,1)-cluster_height))
    col_start = Numbers.rand(rng, 1:(size(world_map,2)-cluster_width))

    for row = row_start:row_start+cluster_height
        for col = col_start:col_start+cluster_width
            world_map[row][col] = true
        end
    end
end

function perc_obstruct(world_map::Array{Bool,2})
    cum_sum = 0.0
    for i in indices(world_map,1)
        for j in indices(world_map,2)
            cum_sum += world_map[i][j]
        end
    end
    return cum_sum/(size(world_map,1)*size(world_map,2))
end

function initialize_map(world_map::Array{Bool,2}, perc_obstruct::Float, rng::AbstractRNG)

    while perc_obstruct(s.world_map) < perc_obstruct
        make_cluster(s.world_map, rng::AbstractRNG)
    end
end

rng = MersenneTwister(1234)
UAVpomdp() = UAVpomdp(20, falses(20,20),(1,1),(20,20),[s1::LineSensor,s2::CircularSensor], [1.0,1.0,1.0])
initialize_map(UAVpomdp.true_map, 0.5, rng)

grid_size = UAVpomdp.map_size

tk_root = tk.Tk()
canvas = tk.Canvas(tk_root, width=grid_size, height=grid_size)
canvas[:grid](row=0,column=0)
canvas_grid = [[canvas[:create_rectangle](col, row, col+10, row+10, fill='black') 
                for col in 1:grid_size:10]
                for row in 1:grid_size:10]
simulator = SimulatorState(0.2, canvas, canvas_grid)

function get_color(ground_truth::Bool)
    if ground_truth
        return 'black'
    else
        return 'white'
end

function update_simulator(p::UAVpomdp, s::State, sim::SimulatorState)
    for row in 1:(p.map_size/10)
        for col in 1:(p.map_size/10)
            ground_truth = s.world_map[row][col]
            canvas.itemconfig(sim.canvas_grid[row][col], fill=get_color(ground_truth))
        end
    end

    aircraft_row, aircraft_col = s.location
    s.canvas[:itemconfig](s.canvas_grid[aircraft_row][aircraft_col], fill='blue')
end

function render_simulator(s::SimulatorState, tk_root::PyObject)
    tk_root[:update]()
    Tasks.sleep(s.animation_rate)
end

curr_state = State(UAVpomdp.start_coords, 0.0, UAVpomdp.true_map)
policy = [Int(DOWN),Int(RIGHT)]

for n = 1:100
	update_simulator(UAVpomdp, simulator)
    render_simulator(simulator, tk_root)
    action = policy[(n % 2)+1]
    curr_state = generate_s(UAVpomdp, curr_state, action, rng)
end 

