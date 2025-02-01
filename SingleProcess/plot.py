# plot.py can create and save 2D and 3D PNGs and automatically create an animation from them using ffmpeg. Additionally, it can plot potential vs kinetic energies. Note: multiprocessing is used to speed up this process.
# How to use (run python plot.py -help for more info):
# - Please add the full path to the "DD-MM-YYYY_HH-MM-SS_all_positions.npy" file as the first argument.')
# - Then, specify plotting options using the following flags:
#  - "-2d" for 2D Plot + animation
#  - "-3d" for 3D Plot + animation
#  - "-e" for energies plot 
#  - "-s" for combined plot of 4 snapshots.')
# Example for creating 2D animation and plotting energies:  python plot.py output/26-11-2024_22-12-48_all_positions.npy -2d -e


import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from multiprocessing import Pool,cpu_count
from scipy.spatial.distance import pdist, squareform


M_earth = 5.972e24  #kg
R_earth = 6.371e6   #m

R_sun = 6.9634e8    #m

def plot(chunk, scale_dist, dt, scale_time, date, start_index):

    for i, coords in enumerate(chunk):
        global_index = start_index + i
        x, y, z = coords.T

        # Convert to * scale dist
        x = (x * scale_dist) / 3.086e19
        y = (y * scale_dist) / 3.086e19
        z = (z * scale_dist) / 3.086e19

        # Plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=0.5, c='white', marker='.',edgecolors='none')
        ax.set_aspect('equal', 'box')
        
        # Set plot limits and title
        limit = (5 * scale_dist) / 3.086e19
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.title.set_text(f'Snapshot {global_index}. Time: {(((global_index * dt * scale_time) * 3.17098e-8) / 1e6):.1f}Myr')
        
        # Format index as a 4-digit string
        index_str = str(global_index).zfill(4)
        
        # Save PNG snapshot
        fig.tight_layout()
        name = f'plot/{date}/snapshot_{index_str}.png'
        fig.savefig(name, dpi=400)
        plt.close()

def plot_3d(chunk, scale_dist, dt, scale_time, date, start_index):

    for i, coords in enumerate(chunk):
        global_index = start_index + i
        x, y, z = coords.T

        # Convert to * scale dist
        x = (x * scale_dist) / 3.086e19
        y = (y * scale_dist) / 3.086e19
        z = (z * scale_dist) / 3.086e19

        # Plot
        plt.style.use('dark_background')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y,z, s=0.5, c='white', marker='.',edgecolors='none')
        ax.set_aspect('equal', 'box')
        
        # Set plot limits and title
        limit = (4 * scale_dist) / 3.086e19
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.view_init(elev=45, azim=105)
        ax.title.set_text(f'Snapshot {global_index}. Time: {(((global_index * dt * scale_time) * 3.17098e-8) / 1e6):.1f}Myr')
        plt.axis('off')

        # Format index as a 4-digit string
        index_str = str(global_index).zfill(4)
        
        # Save PNG snapshot
        fig.tight_layout()
        name = f'plot/{date}/snapshot_3d_{index_str}.png'
        fig.savefig(name, dpi=400)
        plt.close()

def plot_row(positions, scale_dist, dt, scale_time, date,index):
    fig, ax = plt.subplots(1,4,sharey=True)
    fig.set_figheight(4)
    fig.set_figwidth(16)
    #plt.style.use('dark_background')

    for i in range(4):
        x,y,z = positions[i].T
        
        # Convert to kpc
        x = (x * scale_dist) / 3.086e19
        y = (y * scale_dist) / 3.086e19
        z = (z * scale_dist) / 3.086e19

        ax[i].text(-7,6,f'{index[i]}',c='r',fontsize='20')
        ax[i].set_xticks([-8,-4,0,4,8])

        if i==0:
            ax[i].scatter(x,y, s=2, c='white', marker='.',edgecolors='none')
            ax[i].set_facecolor('black')
            ax[i].set_aspect('equal', 'box')
            ax[i].set_yticks([-8,-4,0,4,8])

            # Set plot limits and title
            limit = (4 * scale_dist) / 3.086e19
            ax[i].set_xlim([-limit, limit])
            ax[i].set_ylim([-limit, limit])


        else:
            ax[i].scatter(x,y, s=2, c='white', marker='.',edgecolors='none')
            ax[i].set_facecolor('black')
            ax[i].set_aspect('equal', 'box')

            # Set plot limits and title
            limit = (4 * scale_dist) / 3.086e19
            ax[i].set_xlim([-limit, limit])
            ax[i].set_ylim([-limit, limit])

    fig.tight_layout()
    name = f'plot/{date}/row_snapshots.pdf'
    fig.savefig(name, dpi=800)
    plt.close()
            

def main(pos_file,vel_file,bool_2d,bool_3d,bool_energies,bool_snaps):
    """ Plot simulation results given certain bool flags """
    all_coords = np.load(pos_file)
    all_velocities = np.load(vel_file)


    # Header info
    info = open(f"{pos_file[:-18]}.log", "r")
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

    print(f'''File = {pos_file}
date = {date}
N = {num_particles}
num steps = {num_steps}
scale_dist = {scale_dist/3.086e19}kpc
galaxy_mass = {galaxy_mass}kg
scale_time = {scale_time}s
galaxy separation = {separation} (m)
collision angle = {collision_angle}  (deg)
time_to_impact = {time_to_impact}  (s)
softening length (scaled) = {softening_length}  (scale_dist)
dt (scaled) = {dt} (scale_time)
save_interval = {save_interval}
sim time = {time_taken}s
num processes = {num_processes}
num np threads = {num_np_threads}
num omp threads = {num_omp_threads}
type = {type}

''')

    if not os.path.exists(f'plot/{date}/'):
        os.makedirs(f'plot/{date}/')
        print(f'plot/{date}/ created')
    else:
        print(f'plot/{date}/ already exists')

    num_savedsteps = int(num_steps/save_interval)

    ### Snapshot Plot ###
    if bool_snaps:
        snapshots = np.linspace(0,num_savedsteps-(3/4 * num_savedsteps),4).round().astype(int)
        # or custom:
        #snapshots = [0,140,180,1000]
        pos_snaps = all_coords[snapshots]
        index = [snapshot*save_interval for snapshot in snapshots] 
        print(f"Creating row plot of snapshots: {snapshots}. Snapshot timesteps are: {index}")
        plot_row(pos_snaps, scale_dist, dt, scale_time, date, index)
        print('Done\n')


    # # Split the array into chunks for each process
    num_cores = cpu_count()
    num_processes = 1
    for N in range(num_cores, 0, -1):  # Start from cpu_count() and go down to 1
        if num_steps % N == 0:
            num_processes = N
            break

    assert num_steps % num_processes == 0
    assert num_processes <= num_cores
    timestep_chunks = np.array_split(all_coords, num_processes)

    ## 2D animation ## 
    if bool_2d:
        # Start multiprocessing
        print(f"Creating 2D PNGs with {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            pool.starmap(plot, [
                (chunk, scale_dist, dt, scale_time, date, i * len(chunk))
                for i, chunk in enumerate(timestep_chunks)
            ])

        name_convention = f'plot/{date}/snapshot_%04d.png'
        print('ffmpeg -r 20 -s 2560x1920 -i '+name_convention+' -vcodec libx264 -pix_fmt yuv420p '+date+'.mp4')
        os.system(f'ffmpeg -r 20 -s 2560x1920 -i {name_convention} -vcodec libx264 -pix_fmt yuv420p plot/{date}/movie.mp4')
        print(f'plot/{date}/movie.mp4 saved')
        print('\n')

    ### 3D animation ###
    if bool_3d:
        # Start multiprocessing
        print(f"Creating 3D PNGs with {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            pool.starmap(plot_3d, [
                (chunk, scale_dist, dt, scale_time, date, i * len(chunk))
                for i, chunk in enumerate(timestep_chunks)
            ])

        name_convention = f'plot/{date}/snapshot_3d_%04d.png'
        print('ffmpeg -r 20 -s 2560x1920 -i '+name_convention+' -vcodec libx264 -pix_fmt yuv420p plot/'+date+'/movie_3d.mp4')
        os.system(f'ffmpeg -r 20 -s 2560x1920 -i {name_convention} -vcodec libx264 -pix_fmt yuv420p plot/{date}/movie_3d.mp4')
        print(f'plot/{date}/movie_3d.mp4 saved')



    ### Energies ###
    if bool_energies:
        print("\nNow plotting energies...")
        masses = np.ones(num_particles)*galaxy_mass/num_particles
        all_velocities = np.array([vel * (scale_dist/scale_time) for vel in all_velocities])    # convert to SI
        squared_velocities_all_timesteps = np.sum(all_velocities**2, axis=2)  # Shape: (num_savedsteps, num_particles)
        
        KE_per_particle_all_timesteps = np.zeros_like(squared_velocities_all_timesteps)
        for i in range(squared_velocities_all_timesteps.shape[0]):
            # at each timestep
            for j in range(squared_velocities_all_timesteps.shape[1]):
                # calculate particle's KE
                KE_per_particle_all_timesteps[i,j] = 0.5 * masses[j] * squared_velocities_all_timesteps[i,j]

        KE_all_timesteps = np.sum(KE_per_particle_all_timesteps,axis=1)
        # print(KE_all_timesteps)

        mass_outer_product = np.outer(masses, masses)
        all_coords_scaled = all_coords * scale_dist  # Scale positions for all timesteps
        potential_energy = np.zeros(num_savedsteps)
        G = 6.67e-11
        try:
            potential_energy = np.load(f'plot/{date}/potential_energy.npy')
        except Exception as err:
            print(err)
            # Compute pairwise distances and energies for all timesteps
            for t in range(0,num_savedsteps):
                # Get pos array for that timestep
                pos = all_coords_scaled[t]  # Shape: (N, 3)
                
                distances = pdist(pos)
                
                distances_square = squareform(distances)

                # Avoid division by zero for self-distances
                np.fill_diagonal(distances_square, np.inf)

                pairwise_pe = -G * mass_outer_product / distances_square  # Shape: (N, N)
        
                # Sum the potential energy for this timestep, divide by 2 to avoid double-counting
                potential_energy[t] = np.sum(pairwise_pe) / 2

            np.save(f'plot/{date}/potential_energy.npy',potential_energy)
            print(f'plot/{date}/potential_energy.npy saved!')
            


        #print(potential_energy)
        x = [*range(num_steps)][::save_interval]
        if len(x)!=len(KE_all_timesteps):
            KE_all_timesteps = KE_all_timesteps[:-1]

        error = f'{len(x)}, {len(potential_energy)}, {len(KE_all_timesteps)}'
        assert len(x) == len(potential_energy) == len(KE_all_timesteps), error
        # Plot energies
        plt.figure(figsize=(6.66666, 4))
        plt.plot(x,KE_all_timesteps, label="Kinetic Energy", color='b')
        plt.plot(x,potential_energy, label="Potential Energy", color='r')
        plt.plot(x,KE_all_timesteps + potential_energy, label="Total Energy", color='g', linestyle='--')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(50,50))
        plt.locator_params(axis='x', nbins=20) 
        plt.xlabel("Timestep")
        plt.ylabel("Energy [J]")
        #plt.title("Kinetic, Potential, and Total Energy over Time")
        plt.legend()
        plt.grid()
        # Save figure
        name = f'plot/{date}/energies.png'
        plt.savefig(name, dpi=400)
        print(f'plot/{date}/energies.png saved')
        plt.close()
   
if __name__ == "__main__":
    args = sys.argv

    if '-help' in args:
        print('plot.py can create and save 2D and 3D PNGs and automatically create an animation from them using ffmpeg. Additionally, it can plot potential vs kinetic energies. Note: multiprocessing is used to speed up this process.')
        print('To use plot.py, please add the path to the "path/to/output/DD-MM-YYYY_HH-MM-SS_all_positions.npy" file as the first argument.')
        print('Then, specify plotting options using the following flags:')
        print('\n"-2d" for 2D Plot + animation, \n"-3d" for 3D Plot + animation, \n"-e" for energies plot, \n"-s" for combined plot of 4 snapshots, \n"-a" to do all of the above.')
        print('\nExample for creating 2D animation and plotting energies:  python plot.py output/26-11-2024_22-12-48_all_positions.npy -2d -e')
        sys.exit()

    try:
        pos_file = str(args[1])
    except Exception as err:
        print('Error: ',err)
        print('Please add the path to the "path/to/output/DD-MM-YYYY_HH-MM-SS_all_positions.npy" file as the first argument.')
        sys.exit()

    bool_2d=False
    bool_3d=False
    bool_energies=False
    bool_snaps=False
    print('Plot options selected:')
    if '-s' in args:
        bool_snaps=True
        print('- Combined plot of 4 snapshots')
    if '-2d' in args:
        bool_2d=True
        print('- 2D Plot + animation')
    if '-3d' in args:
        bool_3d=True
        print('- 3D Plot + animation')
    if '-e' in args:
        bool_energies=True
        print('- Energies Plot (WARNING: this calculates potential for all particles over whole sim, so could take a while for large N!)')
    if '-a' in args:
        bool_2d=True
        bool_3d=True
        bool_energies=True
        bool_snaps=True
        print('- Combined plot of 4 snapshots')
        print('- 2D Plot + animation')
        print('- 3D Plot + animation')
        print('- Energies Plot (WARNING: this calculates potential for all particles over whole sim, so could take a while for large N!)')

    if bool_2d==False and bool_3d==False and bool_energies==False and bool_snaps==False:
        print('Error: none selected. \nPlease indicate by flagging "-2d" for 2D Plot + animation, "-3d" for 3D Plot + animation, "-e" for energies plot, and/or "-s" for combined plot of 4 snapshots.\nOr, provide "-a" to do all of the above.')
        sys.exit()
    print('\n')
    vel_file = f'{pos_file[:-18]}_all_velocities.npy'
    main(pos_file,vel_file,bool_2d,bool_3d,bool_energies,bool_snaps)
