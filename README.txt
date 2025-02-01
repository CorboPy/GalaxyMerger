My GalaxyMerger code presents direct gravitational N-body simulations in the context of simple galaxy mergers. I include three serial programs (Python, Cython, and an unfinished Barnes Hut approximation*), an OpenMP program (via Cython), and two MPI programs (pure Python, and OpenMP via Cython). The MPI Cython script allows for hybrid OpenMP + MPI parallelism, which I explore in detail on BlueCrystal in my report. I will include detailed instructions below, but know that there are also instructions and program info at the top of every script.
If you would like to go straight to a nice animation, I have included the animation of the simulation I display in my report (two N=5000 galaxies), which can be found in GalaxyMerger/example/. This was done on BlueCrystal, and took 1.5 hours across 8 nodes.

*I stopped working on this when I realised how painful it is to parallelize, but I thought I'd include it anyway as it does (technically) work.



## DEPENDENCIES ##
A list of all dependencies can be found in the environment.yml file, and the environment 'venv' can be reconstructed in conda using:

>> conda env create -f environment.yml

This is very similar, and practically identical to, the Python build on BlueCrystal.



## COMPILING CYTHON ##
Cython extension building is required in all three SingleProcess/, OpenMP/, and MPI/. These are to be compiled in the usual way, and I have included setup scripts specific to each folder. 
E.g., to create the C extension for MPI, do:

>> cd GalaxyMerger/MPI/
>> python setup.py build_ext -fi

Repeat this for SingleProcess/ and OpenMP/. These will create the leapfrog.c extension for each folder. Inside each folder, you may also find Cython annotation .htmls for each extension.



## SIMULATIONS OF MERGERS ##
I have created three relaxed initial conditions of galaxy mergers with N=800 and N=2000 ready for you to use straight away. You can find these in the folders initial_galaxy_800/ and initial_galaxy_2000/. I have included plots so you can get an idea of the setup. If you're interested in making and relaxing your own galaxies, see the following section on "making and relaxing galaxies".

# SERIAL # 
To run any of the serial codes, you must run them from within GalaxyMerger/SingleProcess/ and supply them with the full or relative path to one of the folders containing the initial conditions. If you wish to change parameters like the total number of timesteps, the number of saved timesteps, the step size, or the value for the softening parameter, these can be found hard-coded under the main() function in each program (except Barnes-Hut - here it can be found under if __name__ == "__main__":).
E.g. to run the serial program that utilizes the C extension on the N=800 initial conditions, do:

>> cd GalaxyMerger/SingleProcess/
>> python N_body_Single_Cython.py ../initial_galaxy_800

You should see the simulation parameters printed first before the code enters the main loop. If successful, the positions, velocities, and a .log file containing simulation info, will be saved to SingleProcess/output. To animate the results, see the subsection on "plotting and animating".

# OPENMP # 
To run the OpenMP program, you must run them from within GalaxyMerger/OpenMP/ and supply them with the full or relative path to one of the folders containing the initial conditions. If you wish to change parameters like the total number of timesteps, the number of saved timesteps, the step size, or the value for softening parameter, these can be found hard-coded under the main() function in each program. If you wish to change the number of OpenMP threads, these can be found a few lines into the code as the variable 'num_omp_threads', and I have set this = 8 as a default.
E.g. to run the OpenMP program on the N=800 initial conditions, do:

>> cd GalaxyMerger/OpenMP/
>> python N_body_OpenMP.py ../initial_galaxy_800

You should see the simulation parameters printed first before the code enters the main loop. If successful, the positions, velocities, and a .log file containing simulation info, will be saved to OpenMP/output. To animate the results, see the subsection on "plotting and animating".

# MPI + OPENMP # 
To run the MPI programs, you must run them from within GalaxyMerger/MPI/ and supply them with the full or relative path to one of the folders containing the initial conditions. If you wish to change parameters like the total number of timesteps, the number of saved timesteps, the step size, or the value for the softening parameter, these can be found hard-coded under the main() function in each program. For N_body_MPI_Cython.py, if you wish to change the number of OpenMP threads, these can be found a few lines into the code as the variable 'num_omp_threads', and I have set this = 2 as a default.
E.g. to run the MPI Cython program on the N=800 initial conditions over 8 processes, do:

>> cd GalaxyMerger/MPI/
>> mpiexec -np 8 python N_body_MPI_Cython.py ../initial_galaxy_800

You should see the simulation parameters printed first before the code enters the main loop. If successful, the positions, velocities, and a .log file containing simulation info, will be saved to MPI/output. To animate the results, see the next subsection on "plotting and animating".
Note: certain n_p are rejected if it doesn't divide nicely into the number of particles in the system. Generally, n_p = 1, 2, 4, 5, 8, 10, 16, 20, 25 are safe.

# PLOTTING AND ANIMATING #
Inside SingleProcess/, OpenMP/, and MPI/, there is a plot.py script which can create and save 2D and 3D PNGs and automatically create an animation from them using FFmpeg. Additionally, it can plot potential vs kinetic energies, but this can take a while for large N as it has to calculate the potential.
The plot.py script works by saving PNGs of each snapshot from the simulation, before using FFmpeg to create an animation from the PNGs. Depending on how many snapshots are being saved, this can mean you have a lot of PNGs being created!
NOTE: multiprocessing is used to substantially speed up the plotting process.

To run plot.py, it must be ran from the folder that the original simulation was run in. The first argument to plot.py is the full or relative path to the "output/DD-MM-YYYY_HH-MM-SS_all_positions.npy" file. You then must give the script one or more of the following flags:
"-2d" for 2D Plot + animation, 
"-3d" for 3D Plot + animation, 
"-e" for energies plot, 
"-s" for combined plot of 4 snapshots, 
"-a" to do all of the above.

E.g., to create a 2D animation, 3D animation, combined snapshot pdf plot (used in my report), and energies plot (also used in my report) for an MPI simulation, do:

>> cd GalaxyMerger/MPI/
>> python plot.py output/DD-MM-YYYY_HH-MM-SS_all_positions.npy -a

As the program is running, processes are assigned to speedup the plotting process (this should not exceed the number of physical cores on your system). If successful, you should see a new folder DD-MM-YYYY_HH-MM-SS/ inside MPI/plot/. Here, you can view the animations, entitled movie.mp4 and movie_3d.mp4, the snapshot plot, entitled row_snapshots.pdf, and the energies.png plot. Additionally, the potential energies at each step are saved to a .npy file to prevent the need for recalculating every time you wish to edit the plot.



## MAKING AND RELAXING GALAXIES ## 

In GalaxyMerger/ exists the script make_galaxy.py. This creates an approximate disk of N bodies (in simulation units) that must be subsequently relaxed over 20,000+ timesteps. Parameters such as scale radius, mass, etc are hard-coded and can be changed on line 67 in make_galaxy.py
E.g. to create a galaxy of 1000 particles, do:

>> cd GalaxyMerger/
>> python make_galaxy.py 1000

This creates .npy files for mass, positions, and velocities, and saves them in GalaxyMerger/. An info.txt is also created, which contains info on scale radius, scale height, etc.

To relax your galaxy, simply run any of the programs without an argument. Ensure that you are relaxing over a sufficient number of timesteps.
E.g., to relax your galaxy using OpenMP, do:

>> python N_body_OpenMP.py

If successful, the positions, velocities, and a .log file containing simulation info, will be saved to OpenMP/output.
When you are satisfied that your galaxy is sufficiently relaxed, you can now use setup_collision.py in GalaxyMerger/ to setup a collision between two of the same galaxy. Before running, adjust collision parameters accordingly on line 150. If you wish, you can also rotate Galaxy 2 around the x-axis by setting a non-zero theta_rot. 
Next, run setup_collision.py with the positions .npy file from your relaxed simulation as the argument.
E.g., if you relaxed your simulation using OpenMP, do:

>> python setup_collision.py OpenMP/output/DD-MM-YYYY_HH-MM-SS_all_positions.npy

This will save a new set of initial merger conditions to a new folder called initial_galaxy_N, where N is the number of particles in the combined system of galaxies (i.e. 2000 if your initial galaxy was 1000).
Now, you can run your merger simulation using the instructions outlined in the previous section :)