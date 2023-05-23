def convert_time_pyNN_to_nest(sim, offset):
    if sim.simulator.state.t > 0:
        return offset + sim.get_time_step() 
    else:
        return offset
