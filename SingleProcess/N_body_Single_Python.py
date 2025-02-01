# N_body_Single_Python.py is serial Python... and is slow!
# How to use:
#   - Adjust simulation parameters (timestep, num_steps, etc) defined at the start of main()
#   - If want to relax a single galaxy with initial conditions in GalaxyMerger/, run without an argument. E.g. python N_body_Single_Python.py
#   - If want to simualte galaxy merger, run with "../initial_galaxy_N" as an argument. E.g. for N=2000 merger: python N_body_Single_Python.py ../initial_galaxy_2000

import os
num_threads = "1" # Limit all to a single thread
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["OMP_NUM_THREADS"] = num_threads
import numpy as np
import time
import sys
import datetime as dt

now = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

G = 1.0
#G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2

def leapfrog_step(positions, velocities, accels, masses, dt, softening_length):
    """Perform a kick-drift-kick leapfrog step for all particles.

    Args:
        positions (NDArray): positions of all particles 
        velocities (NDArray): masses of all particles 
        accels (NDArray): accelerations of all particles 
        masses (NDArray): velocities of all particles
        dt (float): time step size
        softening_length (float): epsilon - softening parameter for gravitational interactions

    Returns:
        tuple: (updated) local_positions, local_velocities, local_accels (NDArray objects)
    """
    # Kick: update velocities by half timestep
    velocities += 0.5 * accels * dt

    # Drift: update positions by full timestep
    positions += velocities * dt

    # Recalculate accels
    accels = calculate_accels(positions, masses,softening_length)

    # Kick: update velocities by another half timestep
    velocities += 0.5 * accels * dt

    return positions, velocities, accels

def calculate_accels(positions, masses,softening_length):
    """ Compute acceleration due to gravity on all particles due to all other particles. """
    num_particles = len(masses)
    accels = np.zeros_like(positions)
    
    for i in range(num_particles):
        for j in range(num_particles):
            if i!=j:

                # r_vec:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]

                # r squared
                softened_r2 = dx * dx + dy * dy + dz * dz + (softening_length * softening_length)
                softened_r = softened_r2**0.5
                inv_r3 = (softened_r)**(-3)  # 1/r^3 then to be multiplied by (dx,dy,dz). Here we have substituted r_hat for r/mag_r
                
                accels[i, 0] += G * masses[j] * inv_r3 * dx
                accels[i, 1] += G * masses[j] * inv_r3 * dy
                accels[i, 2] += G * masses[j] * inv_r3 * dz

    return accels

def nbody_simulation(num_particles, num_steps, dt,save_interval,softening_length,pos_file,vel_file,mass_file):
    """Performs an N-body simulation on supplied initial conditions.

    Args:
        num_particles (int): Number of particles in the simulation
        num_steps (int): Number of time steps
        dt (float): time step size
        save_interval (int): the interval at which snapshots of the simulation are saved
        softening_length (float): epsilon - softening parameter for gravitational interactions
        pos_file (str): path to .npy positions file
        vel_file (str): path to .npy velocities file
        mass_file (str): path to .npy masses file

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

    # Initial accels array
    accels = np.zeros_like(positions)

    for step in range(num_steps):
        positions, velocities, accels = leapfrog_step(positions, velocities, accels, masses, dt,softening_length)

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

def main():
    try:
        folder = str(sys.argv[1])   # Absoltute path to folder
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

    # Execute N-body simulation
    all_positions, all_velocities, time_taken = nbody_simulation(num_particles, num_steps, dt, save_interval,softening_length,pos_file,vel_file,mass_file)
    
    # Save all positions to a file after the simulation
    print(f"Saving output/{now}_all_positions.npy")
    np.save(f"output/{now}_all_positions.npy", all_positions)

    print(f"Saving output/{now}_all_velocities.npy")
    np.save(f"output/{now}_all_velocities.npy", all_velocities)

    num_processes = num_np_threads = num_omp_threads = 1
    sim_type = 'Single Python'
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
    main()
    