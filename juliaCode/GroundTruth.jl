using PyCall
using DataStructures
include("UAVpomdp.jl")

@pyimport Tkinter as tk

##################################
# Simulator struct and functions #
##################################

struct SimulatorState
	tk_root::PyObject
	canvas::PyObject
	canvas_true_grid::Array{Array{Int64,1},1}
    canvas_belief_grid::Array{Array{Int64,1},1}
    canvas_battery::Array{Int64,1}

    function SimulatorState(grid_world_size::Int64, total_battery::Int64)
        tk_root = tk.Tk()

        grid_tile_size = 15

        adj_canvas_size = grid_world_size*grid_tile_size
        battery_canvas_space = Int(floor(adj_canvas_size/20))*4
        final_width = 2*adj_canvas_size+battery_canvas_space

        canvas = tk.Canvas(tk_root, width=final_width, height=adj_canvas_size)
        canvas[:grid](row=0,column=0)

        ground_truth_start = adj_canvas_size + Int(floor(battery_canvas_space/3))
        ground_truth_end = (2*adj_canvas_size) + Int(floor(battery_canvas_space/3))-1
        canvas_true_grid = [[canvas[:create_rectangle](col, row, col+grid_tile_size, 
                                                  row+grid_tile_size, fill="white") 
                        for col in ground_truth_start:grid_tile_size:ground_truth_end] 
                        for row in 1:grid_tile_size:adj_canvas_size]

        canvas_belief_grid = [[canvas[:create_rectangle](col, row, col+grid_tile_size, 
                                                  row+grid_tile_size, fill="gray50") 
                        for col in 1:grid_tile_size:adj_canvas_size] 
                        for row in 1:grid_tile_size:adj_canvas_size]

        # battery_canvas_start = 2*adj_canvas_size + 2*Int(floor(battery_canvas_space/3)) 
        # battery_canvas_end = 2*adj_canvas_size + 3*Int(floor(battery_canvas_space/3))
        # battery_tile_size = Int(floor(adj_canvas_size/total_battery))

        # canvas_battery = [canvas[:create_rectangle](battery_canvas_start,row,
        #                                             battery_canvas_end,
        #                                             row+battery_tile_size, fill="green") 
        #                   for row in 0:battery_tile_size:adj_canvas_size]

        return new(tk_root, canvas, canvas_true_grid, canvas_belief_grid)#, canvas_battery)
    end
end

function first_update_simulator(sim::SimulatorState, initial_state::State)

    canvas = sim.canvas
    canvas_true_grid = sim.canvas_true_grid    
    world_map = initial_state.world_map
    map_size = size(canvas_true_grid,1)

    for row = 1:map_size
        for col = 1:map_size
            if world_map[row,col]
                tile_belief = 1.0
            else
                tile_belief = 0.0
            end
            canvas[:itemconfig](canvas_true_grid[row][col], fill=get_color(tile_belief))
        end
    end

end

function first_update_simulator(sim::SimulatorState, world_map::BitArray)

    canvas = sim.canvas
    canvas_true_grid = sim.canvas_true_grid    
    map_size = size(canvas_true_grid,1)

    for row = 1:map_size
        for col = 1:map_size
            if world_map[row,col]
                tile_belief = 1.0
            else
                tile_belief = 0.0
            end
            canvas[:itemconfig](canvas_true_grid[row][col], fill=get_color(tile_belief))
        end
    end

end

function update_simulator(sim::SimulatorState, state::State, belief_state::BeliefState)

    canvas_true_grid = sim.canvas_true_grid
    canvas_belief_grid = sim.canvas_belief_grid
    canvas_battery = sim.canvas_battery
    canvas = sim.canvas
    tk_root = sim.tk_root
    bel_world_map = belief_state.bel_world_map
    location = state.location
    total_battery_used = state.total_battery_used

    map_size = size(canvas_belief_grid,1)

    for row = 1:map_size
        for col = 1:map_size
            tile_belief = bel_world_map[row,col]
            canvas[:itemconfig](canvas_belief_grid[row][col], fill=get_color(tile_belief))
        end
    end

    aircraft_row, aircraft_col = location
    canvas[:itemconfig](canvas_true_grid[aircraft_row][aircraft_col], fill="blue")
    canvas[:itemconfig](canvas_belief_grid[aircraft_row][aircraft_col], fill="blue")

    # for level = 0:total_battery_used
    #     canvas[:itemconfig](canvas_battery[Int(level)+1], fill="white")
    # end

    tk_root[:update]()
    #sleep(.5)
end

function update_simulator(sim::SimulatorState, state::MDPState)

    canvas_true_grid = sim.canvas_true_grid
    canvas_belief_grid = sim.canvas_belief_grid
    canvas = sim.canvas
    tk_root = sim.tk_root
    bel_world_map = state.bel_world_map
    location = state.location

    map_size = size(canvas_belief_grid,1)

    for row = 1:map_size
        for col = 1:map_size
            tile_belief = bel_world_map[row,col]
            canvas[:itemconfig](canvas_belief_grid[row][col], fill=get_color(tile_belief))
        end
    end

    aircraft_row, aircraft_col = location
    canvas[:itemconfig](canvas_true_grid[aircraft_row][aircraft_col], fill="blue")
    canvas[:itemconfig](canvas_belief_grid[aircraft_row][aircraft_col], fill="blue")

    tk_root[:update]()
end

function freeze_simulator(sim::SimulatorState)
    
    tk_root = sim.tk_root
    tk_root[:mainloop]()

end

########################
# Grid world functions #
########################

function make_cluster(world_map::BitArray{2}, rng::MersenneTwister)
    const cluster_height = 1
    const cluster_width = 1

    row_start = Base.Random.rand(rng, -cluster_height:size(world_map,1))
    col_start = Base.Random.rand(rng, -cluster_width:size(world_map,2))

    for row = row_start:row_start+cluster_height
        for col = col_start:col_start+cluster_width
            if row > size(world_map,1) || row <= 0 || col > size(world_map,2) || col <= 0
                continue
            end
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

function initialize_map(map_size::Int64, desired_perc::Float64, rng::MersenneTwister)
    world_map = falses(map_size,map_size)
    while perc_obstruct(world_map) < desired_perc
        make_cluster(world_map, rng)
    end
    return world_map
end

function get_color(tile_belief::Float64)
    if tile_belief == 1.0
        return "black"
    elseif tile_belief == 0.0
        return "white"
    end

    gray_num = Int(floor(100 - tile_belief*100))
    return "gray" * string(gray_num)
end

function cost_of_naive(p::UAVpomdp)
    total_cost = 0
    for i = 1:p.map_size
        row1 = i
        col1 = i
        row2 = i
        col2 = i+1

        total_cost -= 2*p.reward_lambdas[1]
        
        if p.true_map[row1, col1] && i != 1
            total_cost -= p.reward_lambdas[4]
        end 
        if i != p.map_size && p.true_map[row2, col2]
            total_cost -= p.reward_lambdas[4]
        end

        total_cost
    end

    total_cost += p.reward_lambdas[5]

    return total_cost
end

function get_neighbors(loc::Array{Int64,1}, visited_nodes::Array{Any,1}, map_size::Int64)
    push!(visited_nodes, loc)
    offsets = [[1,0],[-1,0],[0,1],[0,-1]]
    neighbors = []

    for offset in offsets
        new_loc = [loc[1] + offset[1], loc[2] + offset[2]]
        
        if (new_loc in visited_nodes) || new_loc[1] <= 0 || new_loc[2] <= 0 || 
            new_loc[1] > map_size || new_loc[2] > map_size
            continue
        end

        push!(neighbors, new_loc)
        push!(visited_nodes, new_loc)
    end

    return neighbors
end

function cost_of_oracle(true_map::BitArray, start_coords::Array{Int64,1}, 
                        goal_coords::Array{Int64,1}, reward_lambdas::Array{Float64,1})
    pq = DataStructures.PriorityQueue()
    visited_nodes = []
    pq[goal_coords] = (true_map[goal_coords[1],goal_coords[2]],0)

    while true
        tile = peek(pq)
        tile_loc = tile[1]
        tile_values = tile[2]

        if tile_loc == start_coords
            cost_of_nfz = reward_lambdas[4]*tile_values[1]
            cost_of_dist = reward_lambdas[1]*tile_values[2]
            #print(string(tile_values[1])*","*string(tile_values[2])*"\n")
            return tile_values[1], reward_lambdas[5] - cost_of_nfz - cost_of_dist
        end

        neighbors = get_neighbors(tile_loc, visited_nodes, size(true_map,1))
        for neighbor in neighbors
            new_values = (tile_values[1] + true_map[neighbor[1],neighbor[2]],tile_values[2]+1)
            enqueue!(pq, neighbor, new_values)
        end

        dequeue!(pq)
    end
end

function get_confident_neighbors(loc::Array{Int64,1}, belief_map::Array{Float64,2}, 
                                 visited_nodes::Array{Any,1}, map_size::Int64)

    offsets = [[1,0],[-1,0],[0,1],[0,-1]]
    neighbors = []

    for offset in offsets
        new_loc = [loc[1] + offset[1], loc[2] + offset[2]]

        if (new_loc in visited_nodes) || new_loc[1] <= 0 || new_loc[2] <= 0 || 
            new_loc[1] > map_size || new_loc[2] > map_size || 
            belief_map[new_loc[1],new_loc[2]] == 0.5
            continue
        end

        push!(neighbors, [new_loc,loc])
        push!(visited_nodes, new_loc)
    end

    if length(neighbors) > 0
        push!(visited_nodes, loc)
    end 

    return neighbors
end

function get_neighbors_for_border(loc::Array{Int64,1}, map_size::Int64)
    offsets = [[1,0],[-1,0],[0,1],[0,-1]]
    neighbors = []

    for offset in offsets
        new_loc = [loc[1] + offset[1], loc[2] + offset[2]]
        
        if new_loc[1] <= 0 || new_loc[2] <= 0 || new_loc[1] > map_size || new_loc[2] > map_size
            continue
        end

        push!(neighbors, new_loc)
    end

    return neighbors
end

function get_confidence_border(belief_map::Array{Float64,2}, visited_nodes::Array{Any,1}, map_size::Int64)
    border = []
    for row in 1:map_size
        for col in 1:map_size
            if belief_map[row,col] != 0.5
                conf_neighbors = get_neighbors_for_border([row,col], map_size)

                for neighbor in conf_neighbors
                    if belief_map[neighbor[1],neighbor[2]] == 0.5
                        push!(border, [row,col])
                        break
                    end
                end
            end 
        end
    end
    return unique(border)
end

function optimal_action_given_information(p::UAVpomdp, belief_state::BeliefState)

    #print("\n\n TAKING MOVEMENT ACTION\n")
    #print(belief_state.bel_world_map)
    pq = DataStructures.PriorityQueue()
    visited_nodes = []
    confidence_border = get_confidence_border(belief_state.bel_world_map, visited_nodes, p.map_size)

    if (belief_state.bel_world_map[p.goal_coords[1],p.goal_coords[2]] != .5) && ~(p.goal_coords in confidence_border)
        enqueue!(pq, [p.goal_coords, 0], (0,0))
    end
    #print(string(confidence_border)*"\n")

    for tile in confidence_border
        l1_dist =  p.goal_coords[1]-tile[1] + p.goal_coords[2]-tile[2]
        is_nfz = Int(belief_state.bel_world_map[tile[1],tile[2]] > 0.5)
        push!(visited_nodes, tile)
        enqueue!(pq, [tile, 0], (is_nfz, l1_dist))
    end

    while true
        tile = peek(pq)
        tile_loc = tile[1][1]
        tile_came_from = tile[1][2]
        tile_values = tile[2]

        #print(string(tile)*"\n")

        if tile_loc == belief_state.bel_location
            if tile_came_from == 0
                return 5
                if tile_loc[1] == p.map_size
                    return Int(LEFT)
                elseif tile_loc[2] == p.map_size
                    return Int(DOWN)
                else
                    if rand() < 0.5
                        return Int(LEFT)
                    else
                        return Int(RIGHT)
                    end
                end
            end

            row_change = tile_came_from[1] - belief_state.bel_location[1]
            col_change = tile_came_from[2] - belief_state.bel_location[2]

            if row_change == 0 && col_change == -1
                return Int(LEFT)
            elseif row_change == 0 && col_change == 1
                return Int(RIGHT)
            elseif row_change == 1 && col_change == 0
                return Int(DOWN)
            else
                return Int(UP)
            end
        end

        neighbors = get_confident_neighbors(tile_loc, belief_state.bel_world_map, visited_nodes, p.map_size)
        for neighbor in neighbors
            neighbor_loc = neighbor[1]
            is_nfz = Int(belief_state.bel_world_map[neighbor_loc[1],neighbor_loc[2]] > 0.5)
            new_values = (tile_values[1] + is_nfz, tile_values[2]+1)
            enqueue!(pq, neighbor, new_values)
        end

        dequeue!(pq)
    end
end

function choose_information_action(p::UAVpomdp, belief_state::BeliefState)

    max_change = 0
    best_sensor = 0

    for sensor in SENSORS
        change_confidence = p.sensor_set[sensor].changeConfidence(belief_state.bel_world_map, belief_state.bel_location)
        #print(string(sensor)*","*string(change_confidence)*"\n")

        if change_confidence > max_change
            max_change = change_confidence
            best_sensor = sensor
        end
    end

    if max_change < 4.0
        return 0
    else
        return best_sensor
    end
end

function greedy_information_action(p::UAVpomdp, belief_state::BeliefState)
    action = choose_information_action(p, belief_state)

    if action == 0
        return optimal_action_given_information(p, belief_state)
    else
        return action
    end
end