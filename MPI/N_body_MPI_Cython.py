# N_body_MPI_Cython.py is Cython + MPI + OpenMP. Cython + MPI can be tested by setting num_omp_threads = 1
# How to use:
#   - Adjust simulation parameters (timestep, num_steps, etc) defined at the start of main()
#   - If want to relax a single galaxy with initial conditions in GalaxyMerger/, run without an argument. E.g. for 4 processes:  mpiexec -np 4 python N_body_MPI_Cython.py
#   - If want to simualte galaxy merger, run with "../initial_galaxy_N" as an argument. E.g. for 4 processes and N=2000 merger: mpiexec -np 4 python N_body_MPI_Cython.py ../initial_galaxy_2000
#   - Note: if you want to run a hybrid MPI + OpenMP simulation, change num_omp_threads to > 1 below!

import os
num_np_threads = "1"    # Limit numpy to a single thead
num_omp_threads = "2"
os.environ["MKL_NUM_THREADS"] = num_np_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_np_threads
os.environ["OMP_NUM_THREADS"] = num_omp_threads
from mpi4py import MPI
from leapfrog import leapfrog_step  # Import leapfrog_step.c
import numpy as np
import datetime as dt
import sys
import cProfile

now = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get this worker's number 
size = comm.Get_size()  # Number of ranks (processes)

#G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
G=1

try:
    folder = str(sys.argv[1])   # Absoltute path to initial_galaxy_N/ folder
    pos_file = f'{folder}/init_pos.npy'
    vel_file = f'{folder}/init_vel.npy'
    mass_file = f'{folder}/init_mass.npy'
    info_file = f'{folder}/init_info.txt'
except Exception as err:
    if rank==0:
        print(err,'\nNo folder selected, using initial conditions in GalaxyMerger/')
        sys.stdout.flush()
    pos_file = '../init_pos.npy'
    vel_file = '../init_vel.npy'
    mass_file = '../init_mass.npy'
    info_file = f"../init_info.txt"

def nbody_simulation(num_particles, num_steps, dt, save_interval,softening_length,num_omp_threads,pos_file,vel_file,mass_file):
    """Performs an N-body simulation on supplied initial conditions. Paralleized via OpenMP multithreading.

    Args:
        num_particles (int): Number of particles in the simulation
        num_steps (int): Number of time steps
        dt (float): time step size
        save_interval (int): the interval at which snapshots of the simulation are saved
        softening_length (float): epsilon - softening parameter for gravitational interactions
        num_omp_threads (int): number of threads (OpenMP) to run per MPI process
        pos_file (str): path to .npy positions file
        vel_file (str): path to .npy velocities file
        mass_file (str): path to .npy masses file

    Returns:
        IF RANK 0: tuple: all_positions (NDArray), all_velocities (NDArray), time_taken (float)
        OTHERWISE: (0,0,0)
    """
    # Probably not the most efficient way to do this but oh well
    if rank==0:
        # Rank 0 gets the initial conditions
        pos_init = np.load(pos_file)
        vel_init = np.load(vel_file)
        mass_init = np.load(mass_file)

        # Array of indexes
        indicies = np.arange(0, len(pos_init),dtype=np.int32)
        indicies_to_send = np.split(indicies,size)
    else:
    #     pass
        indicies_to_send = None
        pos_init = None
        vel_init = None
        mass_init = None

    # Distribute indicies to all processes
    local_indicies = comm.scatter(indicies_to_send,root=0)

    # Allocate global arrays
    global_positions = comm.bcast(pos_init, root=0)
    global_velocities = comm.bcast(vel_init, root=0)
    global_masses = comm.bcast(mass_init, root=0)

    if rank == 0:
        start_time = MPI.Wtime()
        num_saved_steps = num_steps // save_interval
        print(f'num_saved_steps = {num_saved_steps}')
        all_positions = np.zeros((num_saved_steps+1, num_particles, 3))   # Init all_positions (+1 for final position)
        all_velocities = np.zeros((num_saved_steps+1, num_particles, 3))   # Init all_positions (+1 for final position)
        save_idx = 0

    local_positions = global_positions[local_indicies]
    local_velocities = global_velocities[local_indicies]
    local_accels = np.zeros_like(local_positions)

    for step in range(num_steps):

        # Gather all positions (all local positions from the other workers) to worker 0
        comm.Gather(local_positions, global_positions, root=0) 

        if (step+1) % save_interval ==0:
            comm.Gather(local_velocities, global_velocities, root=0)  # Sending local_velocities from all workers to global_velocities on root 0
            if rank == 0:
                # Save positions and velocities at intervals
                all_positions[save_idx] = global_positions
                all_velocities[save_idx] = global_velocities
                save_idx += 1
                print(f"Step {step+1} / {num_steps}") # Replacing e.g. step 0 with step 1
                sys.stdout.flush()

        # Broadcast updated global positions to all processes
        comm.Bcast(global_positions, root=0)

        #step_start = MPI.Wtime()
        # Calculate local forces and update positions/velocities
        local_positions, local_velocities, local_accels = leapfrog_step(global_positions,global_masses,local_accels,local_velocities,local_indicies,dt,softening_length,num_omp_threads)

    # Gather final positions to rank 0
    comm.Gather(local_positions, global_positions, root=0)
    comm.Gather(local_velocities, global_velocities, root=0)
    if rank == 0:
        all_positions[save_idx] = global_positions
        all_velocities[save_idx] = global_velocities
        end_time = MPI.Wtime()
        time_taken = end_time - start_time
        print(f"Execution time (MPI): {time_taken} seconds")
        return all_positions, all_velocities, time_taken
    else:
        # All other ranks return nothing
        return(0,0,0)

def main(num_np_threads,num_omp_threads):
    # Parameters
    dt = 0.001 # in units of scale_time
    softening_length = int(1e18)    # in metres

    num_steps = 50000
    num_snaps = 500
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

    if rank==0:
        print(f'Initial galaxies selected: N = {num_particles}, M = {galaxy_mass}kg, scale radius = {scale_dist/3.24078e20}kpc, collision angle = {collision_angle} deg, scale_time = {scale_time}s')
        print(f'Num OpenMP threads = {num_omp_threads}')
        print(f'Num numpy threads = {num_np_threads}')
        print(f'Num processes = {size}')
        sys.stdout.flush()
    
    # Run the simulation
    all_positions, all_velocities, time_taken = nbody_simulation(num_particles, num_steps, dt, save_interval,softening_length,int(num_omp_threads),pos_file,vel_file,mass_file)

    if rank == 0:
        # Save all positions to a file after the simulation
        print(f"Saving MPI/output/{now}_all_positions.npy")
        np.save(f"output/{now}_all_positions.npy", all_positions)

        print(f"Saving MPI/output/{now}_all_velocities.npy")
        np.save(f"output/{now}_all_velocities.npy", all_velocities)

        num_processes = size
        sim_type = 'MPI (+OpenMP) cython'
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
        # sim type


if __name__ == "__main__":
    main(num_np_threads,num_omp_threads)
    #cProfile.run('main(num_np_threads,num_omp_threads)')