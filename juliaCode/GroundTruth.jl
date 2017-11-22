using PyCall
include("UAVpomdp.jl")

@pyimport Tkinter as tk

##################################
# Simulator struct and functions #
##################################

struct SimulatorState
	tk_root::PyObject
	canvas::PyObject
	canvas_grid::Array{Array{Int64,1},1}
    canvas_battery::Array{Int64,1}

    function SimulatorState(grid_world_size::Int64, total_battery::Int64)
        tk_root = tk.Tk()

        grid_tile_size = 10

        adj_canvas_size = grid_world_size*grid_tile_size
        battery_canvas_space = Int(adjusted_canvas_size/20)*3
        final_width = adj_canvas_size+battery_canvas_space

        canvas = tk.Canvas(tk_root, width=final_width, height=adj_canvas_size)
        canvas[:grid](row=0,column=0)

        canvas_grid = [[canvas[:create_rectangle](col, row, col+grid_tile_size, 
                                                  row+grid_tile_size, fill="white") 
                        for col in 1:grid_tile_size:adj_canvas_size] 
                        for row in 1:grid_tile_size:adj_canvas_size]

        battery_canvas_start = adj_canvas_size + Int(battery_canvas_start/3) 
        battery_canvas_end = adj_canvas_size + 2*Int(battery_canvas_start/3)
        battery_tile_size = 5

        canvas_battery = [canvas[:create_rectangle](battery_canvas_start,row,
                                                    battery_canvas_end,
                                                    row+battery_tile_size, fill="green") 
                          for row in 0:battery_tile_size:adj_canvas_size]

        return new(tk_root, canvas, canvas_grid, canvas_battery)
    end
end

function update_simulator(sim::SimulatorState, state::State)

    # canvas_grid::Array{Array{Int64,1},1}, 
    #                       world_map::BitArray{2}, location::Array{Int64,1})
 
    canvas_grid = sim.canvas_grid
    canvas_battery = sim.canvas_battery
    canvas = sim.canvas
    tk_root = sim.tk_root
    world_map = state.world_map
    location = state.location
    total_battery_used = state.total_battery_used

    map_size = size(canvas_grid,1)

    for row = 1:map_size
        for col = 1:map_size
            ground_truth = world_map[row,col]
            canvas[:itemconfig](canvas_grid[row][col], fill=get_color(ground_truth))
        end
    end

    aircraft_row, aircraft_col = location
    canvas[:itemconfig](canvas_grid[aircraft_row][aircraft_col], fill="blue")

    for level = 1:total_battery_used
        canvas[:itemconfig](canvas_battery[level+1], fill="white")
    end

    tk_root[:update]()
end

########################
# Grid world functions #
########################

function make_cluster(world_map::BitArray{2}, rng::MersenneTwister)
    cluster_height = 2
    cluster_width = 2

    row_start = Base.Random.rand(rng, 1:(size(world_map,1)-cluster_height))
    col_start = Base.Random.rand(rng, 1:(size(world_map,2)-cluster_width))

    for row = row_start:row_start+cluster_height
        for col = col_start:col_start+cluster_width
            world_map[row,col] = true
        end
    end
end

function perc_obstruct(world_map::BitArray{2})
    cum_sum = 0.0
    for i in indices(world_map,1)
        for j in indices(world_map,2)
            cum_sum += world_map[i,j]
        end
    end
    return cum_sum/(size(world_map,1)*size(world_map,2))
end

function initialize_map(world_map::BitArray{2}, desired_perc::Float64, rng::MersenneTwister)
    while perc_obstruct(world_map) < desired_perc
        make_cluster(world_map, rng)
    end
end

function get_color(ground_truth::Bool)
    if ground_truth
        return "black"
    else
        return "white"
    end
end

###############
# Main script #
###############

sim = SimulatorState(0.2, canvas, canvas_grid)

rng = Base.Random.MersenneTwister(1235)
initial_map = falses(50,50)
initialize_map(initial_map, 0.3, rng)
state = State([1,1], 0.0, initial_map)

policy = [Int(DOWN),Int(RIGHT)]

while true
	update_simulator(sim, state)
    action = policy[(n % 2)+1]
    state = generate_s(UAVpomdp, state, action, rng)
end 

