# N_body_OpenMP.py is Cython + OpenMP. Number of threads is adjusted by setting num_omp_threads below
# How to use:
#   - Adjust simulation parameters (timestep, num_steps, etc) defined at the start of main()
#   - If want to relax a single galaxy with initial conditions in GalaxyMerger/, run without an argument. E.g. python N_body_OpenMP.py
#   - If want to simualte galaxy merger, run with "../initial_galaxy_N" as an argument. E.g. for N=2000 merger: python N_body_OpenMP.py ../initial_galaxy_2000


import sys
import os
num_np_threads = "1" # Limit numpy to a single thead
num_omp_threads = "8" # Change this
os.environ["MKL_NUM_THREADS"] = num_np_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_np_threads
os.environ["OMP_NUM_THREADS"] = num_omp_threads
from leapfrog import leapfrog_step
import numpy as np
import time
import datetime as dt
import cProfile

now = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

G = 1.0
#G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-

# Main simulation loop
def nbody_simulation(num_particles, num_steps, dt,save_interval,softening_length,pos_file,vel_file,mass_file,num_threads):
    """Performs an N-body simulation on supplied initial conditions. Paralleized via OpenMP multithreading.

    Args:
        num_particles (int): Number of particles in the simulation
        num_steps (int): Number of time steps
        dt (float): time step size
        save_interval (int): the interval at which snapshots of the simulation are saved
        softening_length (float): epsilon - softening parameter for gravitational interactions
        pos_file (str): path to .npy positions file
        vel_file (str): path to .npy velocities file
        mass_file (str): path to .npy masses file
        num_threads (int): number of threads (OpenMP) used to distribute workload at loop-level

    Returns:
        tuple: all_positions (NDArray), all_velocities (NDArray), time_taken (float)
    """
    # Load in initial conditions
    positions = np.load(pos_file)
    velocities = np.load(vel_file)
    masses = np.load(mass_file)
    
    num_saved_steps = num_steps // save_interval
    # Init all_positions and all_velocities
    all_positions = np.zeros((num_saved_steps, num_particles, 3))
    all_velocities = np.zeros((num_saved_steps, num_particles, 3))

    save_idx = 0
    start_time = time.time()

    # Initial accel array
    accels = np.zeros_like(positions)

    for step in range(num_steps):
        positions, velocities, accels = leapfrog_step(positions, velocities, accels, masses, dt,softening_length,num_threads)

        # Save positions at intervals
        if step % save_interval == 0:
            all_positions[save_idx] = positions
            all_velocities[save_idx] = velocities
            save_idx += 1
            print(f"Step {step} / {num_steps}")

    end_time = time.time()
    time_taken = end_time - start_time 
    print(f"Execution time: {time_taken} seconds")
    return all_positions, all_velocities, time_taken

def main(num_np_threads, num_omp_threads):
    try:
        folder = str(sys.argv[1])   # Absoltute path to initial_galaxy_N/ folder
        pos_file = f'{folder}/init_pos.npy'
        vel_file = f'{folder}/init_vel.npy'
        mass_file = f'{folder}/init_mass.npy'
        info_file = f'{folder}/init_info.txt'
        print(folder)
    except Exception as err:
        print(err,'\nNo folder selected, using initial conditions in GalaxyMerger/')
        pos_file = '../init_pos.npy'
        vel_file = '../init_vel.npy'
        mass_file = '../init_mass.npy'
        info_file = f"../init_info.txt"

    # Parameters
    dt = 0.001 # in units of scale_time
    softening_length = int(1e18) # in metres
    num_steps = 500
    num_snaps = 50
    assert num_steps % num_snaps == 0, 'num_snaps must divide into num_steps'
    save_interval = int(num_steps/num_snaps)

    # Unit stuff
    info = open(info_file, "r")
    num_particles = int(info.readline())
    scale_dist = float(info.readline())
    galaxy_mass = float(info.readline())
    scale_time = float(info.readline())
    separation = float(info.readline())
    collision_angle = float(info.readline())
    time_to_impact = float(info.readline()) # seconds
    info.close()

    softening_length = softening_length/scale_dist # Convert softening length to units of scale distance

    print(f'Initial galaxies selected: N = {num_particles}, M = {galaxy_mass}kg, scale radius = {scale_dist/3.24078e20}kpc, collision angle = {collision_angle} deg, scale_time = {scale_time}s')
    print(f'Num omp threads = {num_omp_threads}')
    print(f'Num np threads = {num_np_threads}')
    
    # Execute N-body simulation
    all_positions, all_velocities, time_taken = nbody_simulation(num_particles, num_steps, dt, save_interval,softening_length,pos_file,vel_file,mass_file,int(num_omp_threads))

    # Save all positions to a file after the simulation
    print(f"Saving OpenMP/output/{now}_all_positions.npy")
    np.save(f"output/{now}_all_positions.npy", all_positions)

    print(f"Saving OpenMP/output/{now}_all_velocities.npy")
    np.save(f"output/{now}_all_velocities.npy", all_velocities)

    num_processes = 1
    sim_type = 'OpenMP Cython'
    log = open(f"output/{now}.log", "w")
    log.write(str(now) + '\n' + str(num_particles) + '\n' + str(num_steps) + '\n' + str(scale_dist) + '\n' + str(galaxy_mass) + '\n' + str(scale_time) + '\n' + str(separation) + '\n' + str(collision_angle) + '\n' + str(time_to_impact) + '\n' + str(softening_length) + '\n' + str(dt) + '\n' + str(save_interval) + '\n' + str(time_taken) + '\n' + str(num_processes) + '\n' + str(num_np_threads) + '\n' + str(num_omp_threads) + '\n' + sim_type)
    log.close()

    # Time now
    # N
    # Num steps
    # Scale dist/radius (m)
    # Galaxy mass (kg)
    # Scale time (s)
    # Separation (metres)
    # Collision angle
    # Time to impact
    # Softening length (scale_dist)
    # dt (scale_time)
    # save interval
    # time taken (s)
    # num processes
    # num np threads
    # num omp threads
    # time taken

if __name__ == "__main__":
    main(num_np_threads,num_omp_threads)
    #cProfile.run('main(num_np_threads,num_omp_threads)')