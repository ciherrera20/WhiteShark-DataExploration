import argparse
import os
import yaml
import pandas as pd
# from geopandas import GeoDataFrame, read_file
# from shapely.geometry import Point, LineString, Polygon
from datetime import datetime, timedelta
import movingpandas as mpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import scipy
# import scipy.ndimage as ndi
import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
# import scipy.stats
from common import get_shark_gdf, wrap_to_pi
from tqdm import tqdm

def add_columns(traj):
    # Add a timedelta column
    n = traj.df.shape[0]
    timedeltas = [timedelta()] + [traj.df.index[i] - traj.df.index[i - 1] for i in range(1, n)]
    traj.df['TIMEDELTA'] = timedeltas

    # Add velocities and headings to each trajectory
    traj.add_speed()
    traj.add_direction()

    # Compute turning angles
    def bound_angle_diff(theta_diff):
        return (180 / np.pi) * wrap_to_pi(theta_diff * (np.pi / 180))
    turning_angles = [0] + [bound_angle_diff(traj.df['direction'][i + 1] - traj.df['direction'][i]) for i in range(1, n - 1)] + [0]
    traj.df['turning_angle'] = turning_angles

def plot_state_posterior(ax, state_posterior_probs, observed_data, title, label='turning angles', ylabel='angle (rad)'):
    '''Plot state posterior distributions.'''
    ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | angles)')
    ax.set_ylim(0., 1.1)
    ax.set_ylabel('posterior probability')
    ax2 = ax.twinx()
    ln2 = ax2.plot(observed_data, c='black', alpha=0.3, label=label)
    ax2.set_title(title)
    ax2.set_ylabel(ylabel)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    ax.grid(True, color='white')
    ax2.grid(False)

def main(input, output, num_states, turning_angle=True, speed=True, depth=False):
    # Read data from an input file
    filepath = os.path.normpath(input)
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # Create the save directory if necessary
    savepath = os.path.normpath(os.path.join(output, filename))
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    shark_gdf = get_shark_gdf(filepath)
    traj = mpd.TrajectoryCollection(shark_gdf, 'TRANSMITTER').trajectories[0]
    add_columns(traj)

    # Get number of observations by adding the booleans
    num_obs = turning_angle + speed + depth
    if num_obs == 0:
        raise ValueError('You must have at least one observation')
    
    # Create a message describing the observations used
    obs_msg = ''
    if turning_angle:
        obs_msg += 't'
    if speed:
        obs_msg += 's'
    if depth:
        obs_msg += 'd'

    # Map indices to observation type
    obs_dict = {}
    if turning_angle:
        obs_dict[len(obs_dict)] = 'turning_angle'
    if speed:
        obs_dict[len(obs_dict)] = 'speed'
    if depth:
        obs_dict[len(obs_dict)] = 'depth'

    def get_savename(msg, only_msg=False):
        '''Create the name of a file by appending a message to a string describing the data'''
        if only_msg:
            return os.path.normpath(os.path.join(savepath, '{}'.format(msg)))
        else:
            return os.path.normpath(os.path.join(savepath, '{}s-{}-{}'.format(num_states, obs_msg, msg)))

    # Define a function to save figures
    def savefig(msg, only_msg=False):
        '''Save a matplotlib figure'''
        plt.savefig(get_savename(msg, only_msg=only_msg), bbox_inches='tight')
        plt.close()

    # Plot and save the trajectory
    traj.plot(linestyle='None')
    plt.title('Trajectory')
    savefig('traj', only_msg=True)

    # Create a random number generator
    rng = np.random.default_rng()

    # Plot and save turning angles
    turning_angles = wrap_to_pi(np.radians(np.array(traj.df['turning_angle'])))
    plt.hist(turning_angles, bins=np.linspace(-np.pi, np.pi, 30))
    plt.title('Turning angle histogram')
    plt.xlabel('angle (rad)')
    plt.ylabel('counts')
    savefig('turning-angles', only_msg=True)

    # Plot and save speeds
    speeds = np.array(traj.df['speed'])
    plt.hist(speeds, bins=np.linspace(0, np.ceil(np.max(speeds)), 30))
    plt.title('Speed histogram')
    plt.xlabel('speed (m/s)')
    plt.ylabel('counts')
    savefig('speeds', only_msg=True)

    # Randomly initialize the initial state distribution as well as the transition probabilities
    initial_logits = tf.Variable(rng.random([num_states]), name='initial_logits', dtype=tf.float32)
    transition_logits = tf.Variable(rng.random([num_states, num_states]), name='transition_logits', dtype=tf.float32)
    
    dists = []
    if turning_angle:
        # Initialize locations and concentrations of Von Mises distributions for turning angles
        vm_locs = tfp.util.TransformedVariable(
            initial_value=np.zeros(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Sigmoid(low=-np.pi, high=np.pi),
            name='vm_locs')
        
        vm_cons = tfp.util.TransformedVariable(
            initial_value=np.zeros(num_states, dtype=np.float32) + 1e-2,
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='vm_locs')
        dists.append(tfd.VonMises(loc=vm_locs, concentration=vm_cons))

    if speed:
        # Initialize shapes and rates of Gamma distributions for step length
        gamma_shapes = tfp.util.TransformedVariable(
            initial_value=np.ones(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='gamma_shapes')

        gamma_rates = tfp.util.TransformedVariable(
            initial_value=np.ones(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='gamma_shapes')
        dists.append(tfd.Gamma(concentration=gamma_shapes, rate=gamma_rates))

    # Create the joint distributions
    joint_dists = tfd.Blockwise(dists)

    # Create the HMM
    hmm = tfd.HiddenMarkovModel(
        initial_distribution = tfd.Categorical(logits=initial_logits),
        transition_distribution = tfd.Categorical(logits=transition_logits),
        observation_distribution = joint_dists,
        num_steps = len(turning_angles)
    )

    observations = []
    if turning_angle:
        observations.append(turning_angles)
    if speed:
        observations.append(speeds)
    observations = np.array(observations).T

    # Define a loss function
    def compute_loss():
        return -tf.reduce_logsumexp(hmm.log_prob(observations))

    # Define an optimizer to perform back propagation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=0.01)

    print('Training the hmm:')

    t = tqdm(total=1000)
    
    def trace_fn(traceable_quantities):
        t.update()
        return traceable_quantities.loss

    loss_history = tfp.math.minimize(
        loss_fn=compute_loss,
        num_steps=1000,
        optimizer=optimizer,
        convergence_criterion=criterion,
        trace_fn=trace_fn)
    t.close()

    print('Training finished')
    
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('training steps')
    plt.ylabel('loss function (negative log likelihood)')
    savefig('loss')

    # Plot and save the observation distributions
    fig, axs = plt.subplots(1, num_obs, figsize=(5 * num_obs, 5))
    if type(axs) != np.ndarray:
        axs = np.array([axs])
    num = 1001
    x = np.linspace(-np.pi, np.pi, num).reshape(num, 1)
    for (j, (obs_dist, ax)) in enumerate(zip(hmm.observation_distribution.distributions, axs)):
        y = obs_dist.prob(x).numpy()
        for i in range(y.shape[1]):
            if obs_dict[j] == 'turning_angle':
                label = 'state {}, loc={:.2f}, concentration={:.2f}'.format(i, vm_locs[i], vm_cons[i])
                title = 'Turning angle distributions'
            elif obs_dict[j] == 'speed':
                label = 'state {}, mean={:.2f}, mode={:.2f}'.format(i, gamma_shapes[i] / gamma_rates[i], (gamma_shapes[i] - 1) / gamma_rates[i])
                title = 'Speed distributions'
            ax.plot(x[:, 0], y[:, i], label=label)
            ax.set_title(title)
            ax.legend(loc='upper right')
    savefig('emissions')

    # Infer the posterior distributions
    posterior_dists = hmm.posterior_marginals(observations)
    posterior_probs = posterior_dists.probs_parameter().numpy()

    # Plot and save the posterior distributions
    fig, axs = plt.subplots(num_states, num_obs, figsize=(7 * num_obs, 5 * num_states))
    axs = axs.reshape(num_states, num_obs)
    for state, ax_row in enumerate(axs):
        for i, ax in enumerate(ax_row):
            if obs_dict[i] == 'turning_angle':
                plot_state_posterior(ax, posterior_probs[:, state], turning_angles, 'state {} (mean turning angle {:.2f} rad)'.format(state, vm_locs[state]))
            elif obs_dict[i] == 'speed':
                plot_state_posterior(ax, posterior_probs[:, state], speeds, 'state {} (mean speed {:.2f} m/s)'.format(state, gamma_shapes[state] / gamma_rates[state]), label='speed', ylabel='speed (m/s)')
    savefig('posterior-probs')

    # Plot and save the trajectory with states
    x = [point.coords[0][0] for point in traj.df['geometry']]
    y = [point.coords[0][1] for point in traj.df['geometry']]
    fig, ax = plt.subplots(figsize=(10, 10))
    cmaplist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:num_states]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, num_states)
    color = np.argmax(posterior_probs, axis=1)
    sc = ax.scatter(x, y, c=color, cmap=cmap)
    traj.df['state'] = color
    traj.plot(ax=ax, marker='o', column='state', cmap=cmap)
    ticks = np.array(list(range(num_states)))
    tick_labels = ['state {}'.format(i) for i in range(num_states)]
    cbar = plt.colorbar(sc, fraction=0.03)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    plt.title('Trajectory with states')
    savefig('traj-with-states')
    
    # Save to csv with states
    # traj.df.to_csv('{}-{}'.format(os.path.normpath(os.path.join(savepath, filename)), '{}-states-with-speeds.csv'.format(num_states)))
    traj.df.to_csv(get_savename('states.csv'))

    # Create a file to dump some information about the HMM
    f = open(get_savename('hmm-info'), 'w')
    f.write('Initial probs:\n')
    f.write(str(tf.nn.softmax(initial_logits).numpy()))
    f.write('\n\nTransition probs:\n')
    f.write(str(tf.nn.softmax(transition_logits).numpy()))
    if turning_angle:
        f.write('\n\nVM Locs:\n')
        f.write(str(vm_locs.numpy()))
        f.write('\n\nVM Cons:\n')
        f.write(str(vm_cons.numpy()))
    if speed:
        f.write('\n\nGamma shapes:\n')
        f.write(str(gamma_shapes.numpy()))
        f.write('\n\nGamma rates:\n')
        f.write(str(gamma_rates.numpy()))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit an HMM to the given data')
    parser.add_argument('input', nargs='?', help='CSV file containing a single shark trajectory.')
    parser.add_argument('output', nargs='?', help='Location to save results to. Defaults to \'../graphs\'')
    parser.add_argument('-c', '--config', help='Name of the config yaml file to use. All of the other arguments can be provided there using the long argument name without the starting dashes.')
    parser.add_argument('-n', '--num-states', type=int, help='Number of states to use for the HMM. Defaults to 2.')
    parser.add_argument('-t', '--turning-angle', action='store_false', default=None, help='Whether to incorporate turning-angle measurements. Defaults to True.')
    parser.add_argument('-s', '--speed', action='store_false', default=None, help='Whether to incorporate speed measurements. Defaults to True.')
    parser.add_argument('-d', '--depth', action='store_true', default=None, help='Whether to incorporate depth measurements. Defaults to False.')
    args = parser.parse_args()
    # print(args)

    # Parameter defaults
    defaults = {
        'output': '../graphs',
        'num_states': 2,
        'turning_angle': True,
        'speed': True,
        'depth': False
    }

    # Create parameters dictionary from command line arguments and a config yaml file if provided
    params = vars(args)
    if args.config:
        basepath = os.path.dirname(args.config)
        config = yaml.safe_load(open(args.config, 'r'))
        paths = ['input', 'output']
        for path in paths:
            arg_path = path.replace('-', '_')
            if params[arg_path] is None and path in config:
                params[arg_path] = os.path.normpath(os.path.join(basepath, config[path]))
        opt_args = ['num-states', 'turning-angle', 'speed', 'depth']
        for name in opt_args:
            arg_name = name.replace('-', '_')
            if params[arg_name] is None and name in config:
                params[arg_name] = config[name]
    for key, val in defaults.items():
        if params[key] is None:
            params[key] = val
    print('params:', params)

    # Run the main function
    main(params['input'], params['output'], params['num_states'], turning_angle=params['turning_angle'], speed=params['speed'], depth=params['depth'])