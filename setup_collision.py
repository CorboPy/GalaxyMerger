## setup_collision.py combines the same relaxed galaxy twice to create initial conditions for galaxy merger

# How to use:
#   - First make sure you have relaxed your galaxy. This can be done by simply running any of the codes in the usual way, but do not pass initial_galaxy_N/ as an argument to them.
#   - Adjust collision parameters accordingly (line 150)
#   - Run setup_collision.py with the path to the relaxed galaxy positions file as an argument, e.g. python setup_collision.py MPI/output/26-11-2024_00-18-42_all_positions.npy
#   - This will save a new set of initial merger conditions to a folder called initial_galaxy_N , where N is the number of particles in the combined system of galaxies (i.e. 2000 if your initial galaxy was 1000) 

import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot(pos,scale_radius,scale_radius_kpc,total_particles,N):
    """ Plots positions of merger initial conditions in 3D and 2D """
    
    x, y, z = pos.T
    # Convert to Kpc
    x =(x*scale_radius) /3.086e19
    y =(y*scale_radius)/3.086e19
    z =(z*scale_radius) /3.086e19

    pos_unit = 'kpc'
    plt.style.use('dark_background')

    # xy, yz combined plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    label_fontsize = 20
    tick_fontsize = 18
    fig.suptitle(f'N = {total_particles} Initial Merger Conditions',fontsize=label_fontsize)

    ax = axes[0]
    ax.scatter(x, y, s=1, c='white', label='disk', zorder=1, marker='.',edgecolors='none')
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-5 * scale_radius_kpc, 5 * scale_radius_kpc])
    ax.set_ylim([-5 * scale_radius_kpc, 5 * scale_radius_kpc])
    ax.set_xlabel(f'x [{pos_unit}]',fontsize=label_fontsize)
    ax.set_ylabel(f'y [{pos_unit}]',fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax = axes[1]
    ax.scatter(y, z, s=1, c='white', label='disk', zorder=1, marker='.',edgecolors='none')
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-5 * scale_radius_kpc, 5 * scale_radius_kpc])
    ax.set_ylim([-5 * scale_radius_kpc, 5 * scale_radius_kpc])
    ax.set_xlabel(f'y [{pos_unit}]',fontsize=label_fontsize)
    ax.set_ylabel(f'z [{pos_unit}]',fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize) 
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    fig.savefig(f'initial_galaxy_{N}/ICs_xy_yz.png', dpi=400)
    plt.close()


    # 3D plot
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z,s=1,c='white',label='disk',zorder=1,marker='.',depthshade=False,edgecolors='none')
    fig.tight_layout()
    ax.set_xlim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_ylim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_zlim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_aspect('equal','box')
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    ax.view_init(-150, -45)
    plt.axis('off')
    fig.savefig(f'initial_galaxy_{N}/ICs_3D.png',dpi=400)
    plt.close()


# Script starts here
if __name__=='__main__':

    # Get  from arg
    try:
        file_pos = str(sys.argv[1])
    except Exception as err:
        print(err,'\nNo galaxy specified. Please include the path to the relaxed galaxy positions file as an argument e.g. python setup_collision.py MPI/output/26-11-2024_00-18-42_all_positions.npy')
        sys.exit()

    file_vel = f'{file_pos[:-18]}_all_velocities.npy'
    file_log = f'{file_pos[:-18]}.log'
    print('GALAXY SELECTED:\n'+file_pos+'\n'+file_vel+'\n'+file_log+'\n')

    # Header info
    info = open(file_log, "r")
    date = str(info.readline()).replace('\n', '')
    num_particles = int(info.readline())
    num_steps = int(info.readline())
    scale_dist = float(info.readline())     # m
    galaxy_mass = float(info.readline())    # kg
    scale_time = float(info.readline())     # s
    separation = float(info.readline()) # (metres)
    collision_angle = float(info.readline())    # (degrees)
    time_to_impact = float(info.readline())    # (s)
    softening_length = float(info.readline())    # (scale_dist)
    dt = float(info.readline())     # (scale_time)
    save_interval = int(info.readline())   
    time_taken = float(info.readline()) # in seconds
    num_processes = int(info.readline())   # num mpi processes
    num_np_threads = int(info.readline())   # num numpy/intel threads
    num_omp_threads = int(info.readline()) # num OpenMP threads
    type = str(info.readline()) 
    info.close()

    print("Relaxed galaxy info:\n")
    print(f'''File = {file_log}
    date = {date}
    N = {num_particles}
    num steps = {num_steps}
    scale_dist = {scale_dist/3.086e19}kpc
    galaxy_mass = {galaxy_mass}kg
    scale_time = {scale_time}s
    galaxy separation = {separation} (m) - this should be zero in this printout
    collision angle = {collision_angle}  (deg) - this should be zero in this printout
    time_to_impact = {time_to_impact}  (s) - this should be zero in this printout
    softening length (scaled) = {softening_length}  (scale_dist)
    dt (scaled) = {dt} (scale_time)
    snapshot save interval = {save_interval}
    execution time = {time_taken}s
    num processes = {num_processes}
    num np threads = {num_np_threads}
    num omp threads = {num_omp_threads}
    simulation type = {type}
    ''')

    # Make sure output files are not from a simulation of two galaxies
    assert separation==collision_angle==time_to_impact==0, f'Simulation output identified as containing two galaxies. Creating a system of >2 galaxies is not currently supported. Please choose the output from a single galaxy relaxation.'

    # Load in the files
    all_pos = np.load(file_pos)
    all_vels = np.load(file_vel)

    # Get the final positions and velocities of the simulation
    pos = all_pos[-1]
    vel = all_vels[-1]
    assert len(pos) == len(vel) == num_particles, f'Length of position array: {len(pos)} and velocity array: {len(vel)} do not match the number of particles: {num_particles}. This should not happen ever.'
    N = 2* num_particles    # N is the number of particles in the system of the two galaxies



    ##~ Collision parameters ~##
    separation_m = 10 * 3.24078e19 # in m
    separation = separation_m/ scale_dist  # convert to units of scale_radius
    time_to_impact_s = 1.57788e15  # 50MYrs in s - but this often isn't accurate and may take some playing around with (see reduce speed)
    theta = 35 * math.pi/180  # Here, theta is the angle between the x separation and y separation. If theta = 0, galaxies will be positioned along x axis
    theta_rot = 0           # Here, theta_rot is the angle in degrees that the second galaxy will be rotated around the x axis. This is to allow for collisions of disks in different planes. theta_rot = 0 will not rotate galaxy 2
    recentre = True            # If True, galaxy is recentered on (x,y,z) = (0,0,0)
    speed_m_s = separation_m / time_to_impact_s
    speed = speed_m_s * (scale_time/scale_dist)
    speed *= 0.1 # Reduce or increase speed if needed



    # Recentre pos at the origin (useful in case galaxy has shifted away from origin)
    if recentre:
        mean_pos = np.mean(pos,axis=0,dtype=np.float64)
        pos[:,0] -= mean_pos[0]
        pos[:,1] -= mean_pos[1]
        pos[:,2] -= mean_pos[2]
        print('Input galaxy has been recentered on (0,0,0). In reality: ',np.mean(pos, axis=0,dtype=np.float64))

    # xsep, ysep (0.5 factor is to position equally far away from the origin)
    x_sep= 0.5*math.cos(theta) * separation 
    y_sep = 0.5*math.sin(theta) * separation

    # xspeed, yspeed
    x_speed = y_speed = 0.5 * math.sqrt(2)/2 * speed  # velocities are 45 degrees. A better solution would be to implement theta
    # print(f'Speed of each galaxy = {speed_m_s} m/s?')
    # print(f'x speed of each galaxy = { 0.5 * math.sqrt(2)/2 * speed_m_s} m/s?')

    # Set up arrays
    num_particles_total = 2*num_particles
    init_pos = np.zeros((num_particles_total,3))
    init_vel = np.zeros((num_particles_total,3))
    init_mass = np.ones(num_particles_total) * (1 / num_particles)    # Each star has a mass equal to total mass of galaxy (=1 in these units) divided by no. of particles

    # Add Galaxy 1 to init_pos
    init_pos[0:num_particles,0] = pos[:,0]
    init_pos[0:num_particles,1] = pos[:,1]
    init_pos[0:num_particles,2] = pos[:,2]

    # Add Galaxy 2 to init_pos
    init_pos[num_particles:,0] = pos[:,0] 
    init_pos[num_particles:,1] = pos[:,1] 
    init_pos[num_particles:,2] = pos[:,2]

    # Rotate galaxy 2 by theta_rot degrees around x axis
    theta_rot = np.deg2rad(theta_rot)
    rotation_matrix = np.array([
        [1,0,0],
        [0, np.cos(theta_rot), -np.sin(theta_rot)],
        [0, np.sin(theta_rot) , np.cos(theta_rot)]
        ])
    #print(init_pos[num_particles:num_particles+5,:])
    init_pos[num_particles:,:] = np.dot(init_pos[num_particles:,:].copy(), rotation_matrix.T)
    init_vel[num_particles:,:] = np.dot(init_vel[num_particles:,:].copy(), rotation_matrix.T)
    #print(init_pos[num_particles:num_particles+5,:])

    # Galaxy 1 Positions separation
    init_pos[0:num_particles,0] -= x_sep
    init_pos[0:num_particles,1] -= y_sep
    init_pos[0:num_particles,2] -= 0

    # Galaxy 2 Positions separation
    init_pos[num_particles:,0] += x_sep
    init_pos[num_particles:,1] += y_sep
    init_pos[num_particles:,2] += 0

    # Galaxy 1 Velocities
    init_vel[0:num_particles,0] = vel[:,0] + x_speed
    init_vel[0:num_particles,1] = vel[:,1] + y_speed
    init_vel[0:num_particles,2] = vel[:,2]

    # Galaxy 2 Velocities
    init_vel[num_particles:,0] = vel[:,0] - x_speed
    init_vel[num_particles:,1] = vel[:,1] - y_speed
    init_vel[num_particles:,2]  = vel[:,2]

    if len(init_pos[:,0])<20:
        print('Positions:\n',init_pos)

    # x offset - useful to centre galaxies on (x,y,z)=(0,0,0)
    offset_x_m = 0 * 3.086e19
    offset_x = offset_x_m/ scale_dist   
    init_pos[:,0] = init_pos[:,0] + offset_x

    # y offset - useful to centre galaxies on (x,y,z)=(0,0,0)
    offset_y_m = 0 * 3.086e19
    offset_y = offset_y_m/ scale_dist   
    init_pos[:,1] = init_pos[:,1] + offset_y

    # print(f'Galaxy 1 particle 1 x velocity = {init_vel[0,0]} = {init_vel[0,0]*(scale_dist/scale_time)} m/s')
    # print(f'Galaxy 1 particle 1 y velocity = {init_vel[0,1]} = {init_vel[0,1]*(scale_dist/scale_time)} m/s')
    # print(f'Galaxy 2 particle 1 x velocity = {init_vel[num_particles,0]} = {init_vel[num_particles,0]*(scale_dist/scale_time)} m/s')
    # print(f'Galaxy 2 particle 1 y velocity = {init_vel[num_particles,1]} = {init_vel[num_particles,1]*(scale_dist/scale_time)} m/s\n')

    # Check initial_galaxy_N folder
    if not os.path.exists(f'initial_galaxy_{N}/'):
        os.makedirs(f'initial_galaxy_{N}/')
        print(f'initial_galaxy_{N}/ created')
    else:
        print(f'initial_galaxy_{N}/ already exists')
    
    # Save to .npy files
    np.save(f'initial_galaxy_{N}/init_pos.npy', init_pos)
    print('\n'+f'initial_galaxy_{N}/init_pos.npy saved')
    np.save(f'initial_galaxy_{N}/init_vel.npy', init_vel)
    print(f'initial_galaxy_{N}/init_vel.npy saved')
    np.save(f'initial_galaxy_{N}/init_mass.npy', init_mass)
    print(f'initial_galaxy_{N}/init_mass.npy saved'+'\n')

    # Save unit info
    info_str = str(num_particles_total) + '\n' + str(scale_dist) + '\n' + str(galaxy_mass) + '\n' + str(scale_time) + '\n' + str(separation_m) + '\n' + str(theta) + '\n' + str(time_to_impact_s)
    info = open(f"initial_galaxy_{N}/init_info.txt", "w")
    info.write(info_str)
    info.close()
    print(info_str)
    # Num particles
    # Scale radius (m)
    # Galaxy mass (kg)
    # Characteristic time (s)
    # Separation (m)
    # Collision angle (rad)
    # Time to impact (s)

    # Plot initial conditions
    scale_dist_kpc = scale_dist / 3.086e19
    plot(init_pos,scale_dist,scale_dist_kpc,num_particles_total,N)