import argparse
from importlib.metadata import distribution
import os
import yaml
import json
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
import tensorflow as tf
# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
# import scipy.stats
from common import get_shark_gdf, wrap_to_pi, mkdir, softplus, iir_filter
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

def plot_state_posterior(ax, state_posterior_probs, observed_data, title, label, ylabel):
    '''Plot state posterior distributions.'''
    ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | obs)')
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

def plot_transition_probs(ax, transition_probs, covariate_data, title, label, ylabel):
    '''Plot state transition probabilities'''
    ln1 = ax.plot(transition_probs, c='blue', lw=3, label='prob')
    ax.set_ylim(0., 1.1)
    ax.set_ylabel('transition probability')
    ax2 = ax.twinx()
    ln2 = ax2.plot(covariate_data, c='black', alpha=0.3, label=label)
    ax2.set_title(title)
    ax2.set_ylabel(ylabel)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    ax.grid(True, color='white')
    ax2.grid(False)

def main(input, output, num_states, observation_types=[], covariate_types=[], overwrite=False):
    # Read data from an input file
    filepath = os.path.normpath(input)
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # Get shark dataframe and turn it into a trajectory
    shark_gdf = get_shark_gdf(filepath)
    traj = mpd.TrajectoryCollection(shark_gdf, 'TRANSMITTER').trajectories[0]
    add_columns(traj)

    # Get number of observation types
    num_obs_types = len(observation_types)
    if num_obs_types == 0:
        raise ValueError('You must have at least one observation')
    
    # Get number of covariate types
    num_cov_types = len(covariate_types)

    # Create a message describing the observation types used
    obs_msg = ''
    if 'turning_angle' in observation_types:
        obs_msg += 't'
    if 'speed' in observation_types:
        obs_msg += 's'
    if 'depth' in observation_types:
        obs_msg += 'd'

    # Create a message describing the covariate types used
    cov_msg = ''
    if 'temperature' in covariate_types:
        cov_msg += 't'
    if 'depth' in covariate_types:
        cov_msg += 'd'

    # Create the subdirectory containing the model information
    if cov_msg != '':
        model_path = '{}s-{}-{}'.format(num_states, obs_msg, cov_msg)
    else:
        model_path = '{}s-{}'.format(num_states, obs_msg)

    # Create the save directory if necessary
    savepath = os.path.normpath(os.path.join(output, shark_gdf['TRANSMITTER'][0], filename))
    mkdir(os.path.join(savepath, model_path))

    # Map indices to observation type
    obs_dict = {}
    if 'turning_angle' in observation_types:
        obs_dict[len(obs_dict)] = 'turning_angle'
    if 'speed' in observation_types:
        obs_dict[len(obs_dict)] = 'speed'
    if 'depth' in observation_types:
        obs_dict[len(obs_dict)] = 'depth'

    def get_savename(msg, no_model_path=False):
        '''Create the name of a file by appending a message to a string describing the data'''
        if no_model_path:
            return os.path.normpath(os.path.join(savepath, '{}'.format(msg)))
        else:
            return os.path.normpath(os.path.join(savepath, model_path, '{}'.format(msg)))

    # Define a function to save figures
    def savefig(msg, no_model_path=False):
        '''Save a matplotlib figure'''
        plt.savefig(get_savename(msg, no_model_path=no_model_path), bbox_inches='tight')
        plt.close()

    # Plot and save the trajectory
    if not os.path.isfile(get_savename('traj', no_model_path=True)) or overwrite:
        print('Creating trajectory plot')
        traj.plot(linestyle='None')
        plt.title('Trajectory')
        savefig('traj', no_model_path=True)
    else:
        print('Trajectory plot already exists')

    # Create a random number generator
    rng = np.random.default_rng()

    # Plot and save turning angles
    turning_angles = wrap_to_pi(np.radians(np.array(traj.df['turning_angle'])))
    turning_angles /= np.array([traj.df['TIMEDELTA'][1].total_seconds()] + [dt.total_seconds() for dt in traj.df['TIMEDELTA'][1:]])
    turning_angles *= min(traj.df['TIMEDELTA'][1:]).total_seconds()
    if not os.path.isfile(get_savename('turning-angles', no_model_path=True)) or overwrite:
        print('Creating turning angle histogram')
        plt.hist(turning_angles, bins=np.linspace(np.min(turning_angles), np.max(turning_angles), 30))
        plt.title('Turning angle histogram')
        plt.xlabel('angular velocity (rad/{}s)'.format(min(traj.df['TIMEDELTA'][1:]).total_seconds()))
        plt.ylabel('counts')
        savefig('turning-angles', no_model_path=True)
    else:
        print('Turning angle histogram already exists')

    # Plot and save speeds
    speeds = np.array(traj.df['speed'])
    if not os.path.isfile(get_savename('speeds', no_model_path=True)) or overwrite:
        print('Creating speed histogram')
        plt.hist(speeds, bins=np.linspace(0, np.ceil(np.max(speeds)), 30))
        plt.title('Speed histogram')
        plt.xlabel('speed (m/s)')
        plt.ylabel('counts')
        savefig('speeds', no_model_path=True)
    else:
        print('Speed histogram already exists')

    # Plot and save depths
    # depths = np.log(np.exp(10**4 * np.array(traj.df['DEPTH'])) + 1) / (10**4)
    if 'depth' in observation_types or 'depth' in covariate_types:
        depths = softplus(np.array(traj.df['DEPTH']), sharpness=1e3)
        if not os.path.isfile(get_savename('depths', no_model_path=True)) or overwrite:
            print('Creating depth histogram')
            plt.hist(depths, bins=np.linspace(0, np.ceil(np.max(depths)), 30))
            plt.title('Depth histogram')
            plt.xlabel('depth (m)')
            plt.ylabel('counts')
            savefig('depths', no_model_path=True)
        else:
            print('Depth histogram already exists')

    # Plot and save temperatures
    if 'temperature' in covariate_types:
        temperatures = np.array(traj.df['Temp_C'])
        temperatures -= np.min(temperatures)
        temperatures = iir_filter(temperatures, ff=0.3)
        if not os.path.isfile(get_savename('temperatures', no_model_path=True)) or overwrite:
            print('Creating temperature histogram')
            plt.hist(temperatures, bins=np.linspace(np.floor(np.min(temperatures)), np.ceil(np.max(temperatures)), 30))
            plt.title('Temperature histogram')
            plt.xlabel('Temperautre (C)')
            plt.ylabel('counts')
            savefig('temperatures', no_model_path=True)
        else:
            print('Temperature histogram already exists')

    # Package together all of the observations
    observations = []
    if 'turning_angle' in observation_types:
        observations.append(turning_angles)
    if 'speed' in observation_types:
        observations.append(speeds)
    if 'depth' in observation_types:
        observations.append(depths)
    observations = np.array(observations).T
    num_observations = len(observations)

    # Randomly initialize the initial state distribution
    initial_logits = tf.Variable(rng.random([num_states]), name='initial_logits', dtype=tf.float32)
    initial_distribution = tfd.Categorical(logits=initial_logits)

    if num_cov_types != 0:
        # Package together all of the covariates
        covariates = []
        if 'temperature' in covariate_types:
            covariates.append(temperatures[1:])
        if 'depth' in covariate_types:
            covariates.append(depths[1:])
        covariates = np.array(covariates).reshape(num_observations - 1, num_cov_types, 1, 1)

        # Randomly intialize the regression coefficients
        # regression_weights = tf.Variable(rng.random([num_cov_types, num_states, num_states]) * (1 - np.diag([1] * num_states)), name='regression_weights', dtype=tf.float32)
        # regression_intercepts = tf.Variable(rng.random([1, num_states, num_states]) * (1 - np.diag([1] * num_states)), name='regression_intercepts', dtype=tf.float32)
        regression_weights = tf.Variable(np.ones([num_cov_types, num_states, num_states]) * (1 - np.diag([1] * num_states)), name='regression_weights', dtype=tf.float32)
        regression_intercepts = tf.Variable(np.zeros([1, num_states, num_states]) * (1 - np.diag([1] * num_states)), name='regression_intercepts', dtype=tf.float32)

        def get_transition_logits():
            return tf.reduce_sum(regression_weights * covariates, axis=1) + regression_intercepts
    else:
        # If there are no covariates, the transition_logits do not depend on time
        transition_logits = tf.Variable(rng.random([num_states, num_states]), name='transition_logits', dtype=tf.float32)

    # Create state-dependent observation distributions
    dists = []
    if 'turning_angle' in observation_types:
        # Initialize locations and concentrations of Von Mises distributions for turning angles
        angle_locs = tfp.util.TransformedVariable(
            initial_value=np.zeros(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Sigmoid(low=-np.pi, high=np.pi),
            name='angle_locs')
        
        angle_cons = tfp.util.TransformedVariable(
            initial_value=np.zeros(num_states, dtype=np.float32) + 1e-2,
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='angle_cons')
        # angle_locs = tfp.util.TransformedVariable(
        #     initial_value=np.array([0.04, 0.19], dtype=np.float32),
        #     bijector=tfp.bijectors.Sigmoid(low=-np.pi, high=np.pi),
        #     name='angle_locs')
        
        # angle_cons = tfp.util.TransformedVariable(
        #     initial_value=np.array([4.03, 0.14], dtype=np.float32),
        #     bijector=tfp.bijectors.Softplus(low=1e-3),
        #     name='angle_cons')
        dists.append(tfd.VonMises(loc=angle_locs, concentration=angle_cons))

    if 'speed' in observation_types:
        # Initialize shapes and rates of Gamma distributions for step length
        speed_shapes = tfp.util.TransformedVariable(
            initial_value=np.ones(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='speed_shapes')

        speed_rates = tfp.util.TransformedVariable(
            initial_value=np.ones(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='speed_rates')
        dists.append(tfd.Gamma(concentration=speed_shapes, rate=speed_rates))

    if 'depth' in observation_types:
        # Initialize shapes and rates of Gamma distributions for step length
        depth_shapes = tfp.util.TransformedVariable(
            initial_value=np.ones(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='depth_shapes')

        depth_rates = tfp.util.TransformedVariable(
            initial_value=np.ones(num_states, dtype=np.float32),
            bijector=tfp.bijectors.Softplus(low=1e-3),
            name='depth_rates')
        dists.append(tfd.Gamma(concentration=depth_shapes, rate=depth_rates))

    # Create the joint distributions
    joint_dists = tfd.Blockwise(dists)

    # Progress bar
    t = tqdm(total=1000)

    hmm = None
    if num_cov_types != 0:
        # Define a loss function
        def compute_loss():
            global hmm
            hmm = tfd.HiddenMarkovModel(
                initial_distribution = initial_distribution,
                transition_distribution = tfd.Categorical(logits=get_transition_logits()),
                observation_distribution = joint_dists,
                num_steps = num_observations,
                time_varying_transition_distribution = True
            )
            return -tf.reduce_logsumexp(hmm.log_prob(observations))
        
        def trace_fn(traceable_quantities):
            # Update progress bar
            t.update()

            # Update regression coefficients
            regression_weights.assign(regression_weights * (1 - np.diag([1] * num_states)))
            regression_intercepts.assign(regression_intercepts * (1 - np.diag([1] * num_states)))
            return traceable_quantities.loss
    else:
        # Create the HMM
        hmm = tfd.HiddenMarkovModel(
            initial_distribution = initial_distribution,
            transition_distribution = tfd.Categorical(logits=transition_logits),
            observation_distribution = joint_dists,
            num_steps = num_observations
        )

        # Define a loss function
        def compute_loss():
            return -tf.reduce_logsumexp(hmm.log_prob(observations))

        def trace_fn(traceable_quantities):
            # Update progress bar
            t.update()
            return traceable_quantities.loss

    # Define an optimizer to perform back propagation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=0.001)

    print('Training the hmm:')

    loss_history = tfp.math.minimize(
        loss_fn=compute_loss,
        num_steps=100,
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

    if num_cov_types != 0:
        # Create the hmm with all of its trained values
        hmm = tfd.HiddenMarkovModel(
            initial_distribution = initial_distribution,
            transition_distribution = tfd.Categorical(logits=get_transition_logits()),
            observation_distribution = joint_dists,
            num_steps = num_observations,
            time_varying_transition_distribution = True
        )
        transition_logits = get_transition_logits()
        transition_probs = tf.exp(transition_logits) / tf.reshape(tf.reduce_sum(tf.exp(transition_logits), axis=2), [num_observations - 1, 2, 1])

    # Plot and save the observation distributions
    fig, axs = plt.subplots(1, num_obs_types, figsize=(5 * num_obs_types, 5))
    if type(axs) != np.ndarray:
        axs = np.array([axs])
    num = 1001
    for (j, (obs_dist, ax)) in enumerate(zip(hmm.observation_distribution.distributions, axs)):
        if obs_dict[j] == 'turning_angle':
            x = np.linspace(-np.pi, np.pi, num).reshape(num, 1)
        elif obs_dict[j] == 'speed':
            x = np.linspace(0, np.ceil(np.max(speeds)), num).reshape(num, 1)
        elif obs_dict[j] == 'depth':
            x = np.linspace(0, np.ceil(np.max(depths)), num).reshape(num, 1)

        y = obs_dist.prob(x).numpy()
        for i in range(y.shape[1]):
            if obs_dict[j] == 'turning_angle':
                label = 'state {}, loc={:.2f}, concentration={:.2f}'.format(i, angle_locs[i], angle_cons[i])
                title = 'Turning angle distributions'
            elif obs_dict[j] == 'speed':
                label = 'state {}, mean={:.2f}, mode={:.2f}'.format(i, speed_shapes[i] / speed_rates[i], (speed_shapes[i] - 1) / speed_rates[i])
                title = 'Speed distributions'
            elif obs_dict[j] == 'depth':
                label = 'state {}, mean={:.2f}, mode={:.2f}'.format(i, depth_shapes[i] / depth_rates[i], (depth_shapes[i] - 1) / depth_rates[i])
                title = 'Depth distributions'
            ax.plot(x[:, 0], y[:, i], label=label)
            ax.set_title(title)
            ax.legend(loc='upper right')
    savefig('emissions')

    # Infer the posterior distributions
    posterior_dists = hmm.posterior_marginals(observations)
    posterior_probs = posterior_dists.probs_parameter().numpy()

    # Plot and save the posterior probabilities
    fig, axs = plt.subplots(num_states, num_obs_types, figsize=(7 * num_obs_types, 5 * num_states))
    axs = axs.reshape(num_states, num_obs_types)
    for state, ax_row in enumerate(axs):
        for i, ax in enumerate(ax_row):
            if obs_dict[i] == 'turning_angle':
                plot_state_posterior(
                    ax =                    ax,
                    state_posterior_probs = posterior_probs[:, state],
                    observed_data =         turning_angles,
                    title =                 'state {} (mean turning angle {:.2f} rad)'.format(state, angle_locs[state]),
                    label =                 'turning angles',
                    ylabel =                'angle (rad)')
            elif obs_dict[i] == 'speed':
                plot_state_posterior(
                    ax =                    ax,
                    state_posterior_probs = posterior_probs[:, state],
                    observed_data =         speeds,
                    title =                 'state {} (mean speed {:.2f} m/s)'.format(state, speed_shapes[state] / speed_rates[state]),
                    label =                 'speed',
                    ylabel =                'speed (m/s)')
            elif obs_dict[i] == 'depth':
                plot_state_posterior(
                    ax =                    ax,
                    state_posterior_probs = posterior_probs[:, state],
                    observed_data =         depths,
                    title =                 'state {} (mean depth {:.2f} m)'.format(state, depth_shapes[state] / depth_rates[state]),
                    label =                 'depth',
                    ylabel =                'depth (m)')
    savefig('posterior-probs')

    if num_cov_types != 0:
        for cov_type, cov_unit, cov_data in zip(['temperature', 'depth'], ['(m)', '(C)'], [temperatures, depths]):
            if cov_type in covariate_types:
                # Plot and save the transition probabilities
                fig, axs = plt.subplots(num_states, num_states, figsize=(7 * num_states, 5 * num_states))
                axs = axs.reshape(num_states, num_states)
                for i, ax_row in enumerate(axs):
                    for j, ax in enumerate(ax_row):
                        plot_transition_probs(
                            ax = ax,
                            transition_probs = transition_probs[:, i, j],
                            covariate_data = cov_data,
                            title = 'Probability S_t = {} given S_t-1 = {}'.format(j, i),
                            label = cov_type,
                            ylabel = '{} {}'.format(cov_type, cov_unit)
                        )
                savefig('transition-probs-over-{}'.format(cov_type))

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
    # traj.df.to_csv(get_savename('states.csv'))

    print('Saving hmm data...')

    # Create a JSON serializable object with the data from this model
    data_obj = {}

    # Save initial probabilities and logits
    data_obj['initial_probs'] = tf.nn.softmax(initial_logits).numpy().tolist()
    data_obj['initial_logits'] = initial_logits.numpy().tolist()
    
    # Save transition probabilities and logits
    if num_cov_types != 0:
        # transition_logits = get_transition_logits()
        # transition_probs = tf.exp(transition_logits) / tf.reshape(tf.reduce_sum(tf.exp(transition_logits), axis=2), [num_observations - 1, 2, 1])
        data_obj['regression_weights'] = regression_weights.numpy().tolist()
        data_obj['regression_intercepts'] = regression_intercepts.numpy().tolist()
    else:
        transition_probs = tf.nn.softmax(transition_logits)
    data_obj['transition_probs'] = transition_probs.numpy().tolist()
    data_obj['transition_logits'] = transition_logits.numpy().tolist()

    # Save posterior probabilities
    data_obj['posterior_probs'] = posterior_probs.tolist()

    # Save states
    data_obj['states'] = color.tolist()

    # Save the state-dependent observation distribution parameters
    data_obj['observation_types'] = []
    obs_dist_params = {}
    if 'turning_angle' in observation_types:
        data_obj['observation_types'].append('turning_angle')
        obs_dist_params['turning_angle'] = {
            'vm_locs': angle_locs.numpy().tolist(),
            'vm_cons': angle_cons.numpy().tolist()
        }
    if 'speed' in observation_types:
        data_obj['observation_types'].append('speed')
        obs_dist_params['speed'] = {
            'gamma_shapes': speed_shapes.numpy().tolist(),
            'gamma_rates': speed_rates.numpy().tolist()
        }
    if 'depth' in observation_types:
        data_obj['observation_types'].append('depth')
        obs_dist_params['depth'] = {
            'gamma_shapes': depth_shapes.numpy().tolist(),
            'gamma_rates': depth_rates.numpy().tolist()
        }
    data_obj['obs_dist_params'] = obs_dist_params
    data_obj['log_likelihood'] = -loss_history[-1].numpy().tolist()

    # Initial probability distribution + state dependent distribution params
    num_params = num_states + (2 * num_obs_types * num_states)
    if num_cov_types != 0:
        # Add in regression coefficients
        num_params += (num_states * (num_states - 1)) * num_cov_types
    else:
        # Add in transition probabilities
        num_params += num_states**2 - num_states
    data_obj['num_params'] = num_params
    data_obj['bayesian_information_criterion'] = num_params * np.log(num_observations) - 2 * data_obj['log_likelihood']

    # # Save the observations used
    # data_obj['observations'] = observations.tolist()

    # Save the covariates used
    data_obj['covariate_types'] = covariate_types
    # if num_cov_types != 0:
    #     data_obj['covariates'] = covariates.tolist()

    # Create a file and dump the model data to it
    f = open(get_savename('hmm-data.json'), 'w')
    f.write(json.dumps(data_obj, indent=4))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit an HMM to the given data')
    parser.add_argument('input', nargs='?', help='CSV file containing a single shark trajectory.')
    parser.add_argument('output', nargs='?', help='Location to save results to. Defaults to \'../graphs\'')
    parser.add_argument('-c', '--config', help='Name of the config yaml file to use. All of the other arguments can be provided there using the long argument name without the starting dashes.')
    parser.add_argument('-n', '--num-states', type=int, help='Number of states to use for the HMM. Defaults to 2.')
    parser.add_argument('-obs', '--observation-types', nargs='*', type=str, help='Name of the observation types to use.')
    parser.add_argument('-cov', '--covariate-types', nargs='*', type=str, help='Name of the covariate types to use.')
    args = parser.parse_args()
    # print(args)

    # Parameter defaults
    defaults = {
        'output': '../graphs',
        'num_states': 2,
        'observation_types': [],
        'covariate_types': []
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
        opt_args = ['num-states', 'observation-types', 'covariate-types']
        for name in opt_args:
            arg_name = name.replace('-', '_')
            if params[arg_name] is None and name in config:
                params[arg_name] = config[name]
    for key, val in defaults.items():
        if params[key] is None:
            params[key] = val
    print('params:', params)

    # Run the main function
    main(params['input'], params['output'], params['num_states'], observation_types=params['observation_types'], covariate_types=params['covariate_types'], overwrite=True)