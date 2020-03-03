#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 05:33:19 2019

@author: Dr Stephen Hanham
based on grating.py

Installation and running:

1. Create and activate a virtual environment
2. pip install numpy scipy matplotlib pandas flatdict
3. pip install gdspy==1.3.2 (seems to be a bug in latest version)
4. export PYTHONPATH='/home/stephen/spins-b_THz_filter'
5. python3 run test1
6. python3 view test1
ISSUE: Getting really bad performance using virtual environment. Stopped using it.

"""

import os
import pickle
import shutil

import gdspy
import numpy as np
from scipy.interpolate import interp1d
from typing import List, NamedTuple, Tuple

# `spins.invdes.problem_graph` contains the high-level spins code.
from spins.invdes import problem_graph
# Import module for handling processing optimization logs.
from spins.invdes.problem_graph import log_tools
from spins.invdes.problem_graph import grid_utils
# `spins.invdes.problem_graph.optplan` contains the optimization plan schema.
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace
#from spins.invdes.problem_graph.functions import stored_energy
from spins.invdes.problem_graph.functions import poynting

# Parameters for optimisation
GRID_SPACING = 40  # Yee cell grid spacing
dx = 40
wg_width = 500
wlen_sim_list = np.linspace(1111.11*1.01, 2000*0.99, 10).tolist()  # List of wavelengths to simulate at
wlen_opt_list = wlen_sim_list  # List of wavelengths to optimise at

lam_low = 1357.47  # Lower wavelength of passband (nm)
lam_upp = 1507.54  # Upper wavelength of passband (nm)

# Silicon refractive index to use for 2D simulations. This should be the
# effective index value.
SI_2D_INDEX = 2.60
# Silicon refractive index to use for 3D simulations.
SI_3D_INDEX = 3.45


def run_opt(save_folder: str) -> None:
    """Main optimization script.

    This function setups the optimization and executes it.

    Args:
        save_folder: Location to save the optimization data.
    """
    os.makedirs(save_folder)

    sim_space = create_sim_space(
        "sim_fg.gds",
        "sim_bg.gds",
        dx=dx
    )
    obj, monitors = create_objective(sim_space)  # or a grating length
    trans_list = create_transformations(obj, monitors, sim_space, cont_iters=150, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)

    # Save the optimization plan so we have an exact record of all the
    # parameters.
    with open(os.path.join(save_folder, "optplan.json"), "w") as fp:
        fp.write(optplan.dumps(plan))

    # Copy over the GDS files.
    shutil.copyfile("sim_fg.gds", os.path.join(save_folder, "sim_fg.gds"))
    shutil.copyfile("sim_bg.gds", os.path.join(save_folder, "sim_bg.gds"))

    # Execute the optimization and indicate that the current folder (".") is
    # the project folder. The project folder is the root folder for any
    # auxiliary files (e.g. GDS files).
    problem_graph.run_plan(plan, ".", save_folder=save_folder)

    # Generate the GDS file.
    # gen_gds(save_folder, sim_width)


def create_sim_space(
        gds_fg_name: str,
        gds_bg_name: str,
        sim_width: float = 8000,  # size of sim_space
        box_width: float = 4000,  # size of our editing structure
        wg_width: float = 500,
        buffer_len: float = 1500,  # not sure we'll need
        dx: int = 40,
        num_pmls: int = 10,
        visualize: bool = False,
) -> optplan.SimulationSpace:
    """Creates the simulation space.

    The simulation space contains information about the boundary conditions,
    gridding, and design region of the simulation.

    Args:
        gds_fg_name: Location to save foreground GDS.
        gds_bg_name: Location to save background GDS.
        etch_frac: Etch fraction of the grating. 1.0 indicates a fully-etched
            grating.
        box_thickness: Thickness of BOX layer in nm.
        wg_width: Width of the waveguide.
        buffer_len: Buffer distance to put between grating and the end of the
            simulation region. This excludes PMLs.
        dx: Grid spacing to use.
        num_pmls: Number of PML layers to use on each side.
        visualize: If `True`, draws the polygons of the GDS file.

    Returns:
        A `SimulationSpace` description.
    """

    # The BOX layer/silicon device interface is set at `z = 0`.
    #
    # Describe materials in each layer.

    # 1) Silicon Nitride

    # Note that the layer numbering in the GDS file is arbitrary. Layer 300 is a dummy
    # layer; it is used for layers that only have one material (i.e. the
    # background and foreground indices are identical) so the actual structure
    # used does not matter.

    # Will need to define out material, just silicon nitride
    # Remove the etching stuff
    # Can define Si3N4 - the material we want to use
    # Fix: try to make multiple layers, but all the same?

    air = optplan.Material(mat_name="air")
    quartz = optplan.Material(index=optplan.ComplexNumber(real=2.10))
    stack = [
        optplan.GdsMaterialStackLayer(
            foreground=quartz,
            background=quartz,
            gds_layer=[100, 0],
            extents=[-10000, -110],  # will probably need to define a better thickness for our layer
        ),
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(index=optplan.ComplexNumber(real=SI_2D_INDEX)),
            background=air,
            gds_layer=[100, 0],
            extents=[-110, 110],  # will probably need to define a better thickness for our layer
        ),
    ]

    mat_stack = optplan.GdsMaterialStack(
        # Any region of the simulation that is not specified is filled with
        # air.
        background=air,
        stack=stack,
    )

    # these define the entire region you wish to scan in the z -direction, not sure for us
    # as we don't require etching or layers
    # will probably change this as thickness may be wrong

    # Create a simulation space for both continuous and discrete optimization.
    simspace = optplan.SimulationSpace(
        name="simspace",
        mesh=optplan.UniformMesh(dx=dx),
        eps_fg=optplan.GdsEps(gds=gds_fg_name, mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds=gds_bg_name, mat_stack=mat_stack),
        # Note that we explicitly set the simulation region. Anything
        # in the GDS file outside of the simulation extents will not be drawn.
        sim_region=optplan.Box3d(
            center=[0, 0, 0],
            extents=[13500, 3350, dx],
        ),
        selection_matrix_type="direct_lattice",  # or uniform
        # PMLs are applied on x- and z-axes. No PMLs are applied along y-axis
        # because it is the axis of translational symmetry.
        pml_thickness=[num_pmls, num_pmls, num_pmls, num_pmls, 0, 0],  # may need to edit this, make z the 0 axis
    )

    if visualize:
        # To visualize permittivity distribution, we actually have to
        # construct the simulation space object.
        import matplotlib.pyplot as plt
        from spins.invdes.problem_graph.simspace import get_fg_and_bg

        context = workspace.Workspace()
        eps_fg, eps_bg = get_fg_and_bg(context.get_object(simspace), wlen=1425)  # edit here

        def plot(x):
            plt.imshow(np.abs(x)[:, 0, :].T.squeeze(), origin="lower")

        plt.figure()
        plt.subplot(3, 1, 1)
        plot(eps_fg[2])
        plt.title("eps_fg")

        plt.subplot(3, 1, 2)
        plot(eps_bg[2])
        plt.title("eps_bg")

        plt.subplot(3, 1, 3)
        plot(eps_fg[2] - eps_bg[2])
        plt.title("design region")
        plt.show()
    return simspace


def create_objective(
        sim_space: optplan.SimulationSpace
) -> Tuple[optplan.Function, List[optplan.Monitor]]:
    """"Creates the objective function."""

    # Create the waveguide source - align with our sim_space
    port1_in = optplan.WaveguideModeSource(
        center=[-6000, 0, 0],  # may need to edit these, not too sure
        extents=[GRID_SPACING, 1500, 600],
        normal=[1, 0, 0],
        mode_num=0,
        power=1.0,
    )

    port1_out = optplan.WaveguideModeOverlap(
        center=[-6000, 0, 0],  # may need to edit these, not too sure
        extents=[GRID_SPACING, 1500, 600],
        normal=[-1, 0, 0],
        mode_num=0,
        power=1.0,
    )

    port2_out = optplan.WaveguideModeOverlap(
        center=[6000, 0, 0],
        extents=[GRID_SPACING, 1500, 600],
        normal=[1, 0, 0],
        mode_num=0,
        power=1.0,
    )

    # Construct the monitors for the metrics and fields
    power_objs = []
    monitor_list = []
    first = True
    obj = 0.
    for wlen in wlen_sim_list:

        epsilon = optplan.Epsilon(
            simulation_space=sim_space,
            wavelength=wlen,
        )

        sim = optplan.FdfdSimulation(
            source=port1_in,
            # Use a direct matrix solver (e.g. LU-factorization) on CPU for
            # 2D simulations and the GPU Maxwell solver for 3D.
            solver="local_direct",
            wavelength=wlen,
            simulation_space=sim_space,
            epsilon=epsilon,
        )

        # Take a field slice through the z=0 plane to save each iteration.
        monitor_list.append(
            optplan.FieldMonitor(
                name="field_{}".format(wlen),  # edit so we can have multiple of same wavelength
                function=sim,
                normal=[0, 0, 1],  # may want to change these normals
                center=[0, 0, 0],
            ))

        if first:
            monitor_list.append(optplan.FieldMonitor(name="epsilon", function=epsilon))
            first = False

        port1_out_overlap = optplan.Overlap(simulation=sim, overlap=port1_out)
        port2_out_overlap = optplan.Overlap(simulation=sim, overlap=port2_out)

        port1_out_power = optplan.abs(port1_out_overlap) ** 2
        port2_out_power = optplan.abs(port2_out_overlap) ** 2

        s11_sim_dB = 10 * optplan.log10(port1_out_power)
        s21_sim_dB = 10 * optplan.log10(port2_out_power)

        power_objs.append(port1_out_power)
        power_objs.append(port2_out_power)

        monitor_list.append(optplan.SimpleMonitor(name="port1_out_power_{}".format(wlen), function=port1_out_power))
        monitor_list.append(optplan.SimpleMonitor(name="port2_out_power_{}".format(wlen), function=port2_out_power))

        wlen_goal, s11_fn, s21_fn = load_s_param_goal()

        if wlen in wlen_opt_list:  # Only optimise for the resonant wavelengths of the ring resonator
            # resonator_energy = stored_energy.StoredEnergy(simulation=sim, simulation_space=sim_space, center=[0, 1000, 0],
            #                                              extents=[2000, 2000, 1000], epsilon=epsilon)

            # monitor_list.append(optplan.SimpleMonitor(name="stored_energy_{}".format(wlen), function=resonator_energy))

            # power_rad_top = poynting.PowerTransmission(field=sim, center=[0, 2500, 0], extents=[2000, 0, 0], normal=[0, 1, 0])
            # power_rad_left = poynting.PowerTransmission(field=sim, center=[-2500, 1000, 0], extents=[0, 2000, 0], normal=[-1, 0, 0])
            # power_rad_right = poynting.PowerTransmission(field=sim, center=[2500, 1000, 0], extents=[0, 2000, 0], normal=[1, 0, 0])

            # monitor_list.append(optplan.SimpleMonitor(name="Pr_top_{}".format(wlen), function=power_rad_top))
            # monitor_list.append(optplan.SimpleMonitor(name="Pr_left_{}".format(wlen), function=power_rad_left))
            # monitor_list.append(optplan.SimpleMonitor(name="Pr_right_{}".format(wlen), function=power_rad_right))

            s11_goal = s11_fn(wlen)  # In power
            s21_goal = s21_fn(wlen)  # In power
            s11_goal_dB = float(10*np.log10(s11_goal))
            s21_goal_dB = float(10*np.log10(s21_goal))

            if lam_low <= wlen <= lam_upp:
                obj += (s21_goal_dB - s21_sim_dB) ** 2 * 10  # Passband objective
            else:
                obj += (s21_goal_dB - s21_sim_dB) ** 2  # Reject band objective

            # Objective function to achieve a particular stored energy
            # Specifying a specific target stored energy for both wavelengths seems to perform better for achieving
            # coupling for multiple resonances
            # obj += (5-resonator_energy)**2

            # Objective to maximise Q of resonator
            # obj += resonator_energy / (power_rad_top + power_rad_left + power_rad_right)

            # Objective to minimise output powers for critical coupling (all power lost in resonator)
            # Update: Did not get good results with this objective fn
            # obj += port1_out_power + port2_out_power

    monitor_list.append(optplan.SimpleMonitor(name="objective", function=obj))
    generate_monitor_spec(monitor_list)
    return obj, monitor_list


def create_transformations(
        obj: optplan.Function,
        monitors: List[optplan.Monitor],
        sim_space: optplan.SimulationSpaceBase,
        cont_iters: int,  # require more to optimise power better
        num_stages: int = 3,
        min_feature: float = 100,
) -> List[optplan.Transformation]:
    """Creates a list of transformations for the device optimization.

    The transformations dictate the sequence of steps used to optimize the
    device. The optimization uses `num_stages` of continuous optimization. For
    each stage, the "discreteness" of the structure is increased (through
    controlling a parameter of a sigmoid function).

    Args:
        opt: The objective function to minimize.
        monitors: List of monitors to keep track of.
        sim_space: Simulation space ot use.
        cont_iters: Number of iterations to run in continuous optimization
            total across all stages.
        num_stages: Number of continuous stages to run. The more stages that
            are run, the more discrete the structure will become.
        min_feature: Minimum feature size in nanometers.

    Returns:
        A list of transformations.
    """
    # Setup empty transformation list.
    trans_list = []

    # First do continuous relaxation optimization.
    # This is done through cubic interpolation and then applying a sigmoid
    # function.
    param = optplan.CubicParametrization(
        # Specify the coarseness of the cubic interpolation points in terms
        # of number of Yee cells. Feature size is approximated by having
        # control points on the order of `min_feature / GRID_SPACING`.
        undersample=3.5 * min_feature / GRID_SPACING,
        simulation_space=sim_space,
        init_method=optplan.UniformInitializer(min_val=0.6, max_val=0.9),
    )

    iters = max(cont_iters // num_stages, 1)
    for stage in range(num_stages):
        trans_list.append(
            optplan.Transformation(
                name="opt_cont{}".format(stage),
                parametrization=param,
                transformation=optplan.ScipyOptimizerTransformation(
                    optimizer="L-BFGS-B",
                    objective=obj,
                    monitor_lists=optplan.ScipyOptimizerMonitorList(
                        callback_monitors=monitors,
                        start_monitors=monitors,
                        end_monitors=monitors),
                    optimization_options=optplan.ScipyOptimizerOptions(
                        maxiter=iters),
                ),
            ))

        if stage < num_stages - 1:
            # Make the structure more discrete.
            trans_list.append(
                optplan.Transformation(
                    name="sigmoid_change{}".format(stage),
                    parametrization=param,
                    # The larger the sigmoid strength value, the more "discrete"
                    # structure will be.
                    transformation=optplan.CubicParamSigmoidStrength(
                        value=4 * (stage + 1)),
                ))
    return trans_list


def generate_monitor_spec(monitors: list):
    """This function automatically generates a monitor spec file. This is a total hack"""
    import yaml

    monitors_list = []
    with open("monitor_spec_dual.yml", 'w') as f:
        f.write('monitor_list:\n')
        for m in monitors:
            if m.name.startswith('field'):
                monitors_list.append(
                    {'monitor_names': f'[{m.name}]', 'monitor_type': 'planar', 'vector_operation': 'magnitude'})
            elif m.name.startswith('port'):
                monitors_list.append(
                    {'monitor_names': f'[{m.name}]', 'monitor_type': 'scalar', 'scalar_operation': 'magnitude'})

        monitors_list.append({'monitor_names': '[epsilon]', 'monitor_type': 'planar', 'scalar_operation': 'magnitude',
                              'vector_operation': 'z'})
        monitors_list.append({'monitor_names': '[objective]', 'monitor_type': 'scalar'})
        yaml.dump(monitors_list, f, default_style=None, default_flow_style=False)

    # Get rid of quotes. This is a really rubbish way of doing it.
    with open("monitor_spec_dual.yml", 'r+') as f:
        text = f.read()
        f.seek(0)
        f.truncate(0)
        f.write(text.replace("'", ""))


def load_s_param_goal():
    """Returns interpolating functions for filter S-params"""
    data = np.loadtxt('filter_resp.csv', delimiter=",")
    wlen = 3e8 / data[:, 0] * 1e6  # Scale to nm
    s11_fn = interp1d(wlen, data[:, 1])
    s21_fn = interp1d(wlen, data[:, 2])

    #import matplotlib.pyplot as plt
    #f = np.linspace(np.amin(data[:, 0]), np.amax(data[:, 0]), 100)
    #plt.figure()
    #plt.plot(data[:, 0], 20*np.log10(data[:, 1]))
    #plt.plot(data[:, 0], 20 * np.log10(data[:, 2]))
    #plt.plot(f, 20*np.log10(s11_fn(f)), ':')
    #plt.plot(f, 20*np.log10(s21_fn(f)), ':')
    #plt.show()

    return wlen, s11_fn, s21_fn


def view_opt(save_folder: str) -> None:
    """Shows the result of the optimization.

    This runs the auto-plotter to plot all the relevant data.
    See `examples/wdm2` IPython notebook for more details on how to process
    the optimization logs.

    Args:
        save_folder: Location where the log files are saved.
    """
    log_df = log_tools.create_log_data_frame(log_tools.load_all_logs(save_folder))
    monitor_descriptions = log_tools.load_from_yml(os.path.join(os.path.dirname(__file__), "monitor_spec_dual.yml"))

    log_tools.plot_monitor_data(log_df, monitor_descriptions, show=False)
    log_tools.plot_power_vs_wlen(log_df, monitor_descriptions, "port1_out", show=False)
    log_tools.plot_power_vs_wlen(log_df, monitor_descriptions, "port2_out", show=False)

    import matplotlib.pyplot as plt
    wlen_goal, s11_fn, s21_fn = load_s_param_goal()
    plt.plot(wlen_goal, 20*np.log10(s21_fn(wlen_goal)), ':r')
    plt.show()


def view_opt_quick(save_folder: str) -> None:
    """Prints the current result of the optimization.

    Unlike `view_opt`, which plots fields and optimization trajectories,
    `view_opt_quick` prints out scalar monitors in the latest log file. This
    is useful for having a quick look into the state of the optimization.

    Args:
        save_folder: Location where the log files are saved.
    """
    with open(workspace.get_latest_log_file(save_folder), "rb") as fp:
        log_data = pickle.load(fp)
        for key, data in log_data["monitor_data"].items():
            if np.isscalar(data):
                print("{}: {}".format(key, data.squeeze()))


def resume_opt(save_folder: str) -> None:
    """Resumes a stopped optimization.

    This restarts an optimization that was stopped prematurely. Note that
    resuming an optimization will not lead the exact same results as if the
    optimization were finished the first time around.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
    """
    # Load the optimization plan.
    with open(os.path.join(save_folder, "optplan.json")) as fp:
        plan = optplan.loads(fp.read())

    # Run the plan with the `resume` flag to restart.
    problem_graph.run_plan(plan, ".", save_folder=save_folder, resume=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("run", "view", "view_quick", "resume"),
        help="Must be either \"run\" to run an optimization, \"view\" to "
             "view the results, \"resume\" to resume an optimization.")
    parser.add_argument("save_folder", help="Folder containing optimization logs.")

    args = parser.parse_args()
    if args.action == "run":
        run_opt(args.save_folder)
    elif args.action == "view":
        view_opt(args.save_folder)
    elif args.action == "view_quick":
        view_opt_quick(args.save_folder)
    elif args.action == "resume":
        resume_opt(args.save_folder)
