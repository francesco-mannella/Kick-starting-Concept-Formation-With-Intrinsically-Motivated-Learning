import numpy as np
task_space = {"xlim": [-10, 50], "ylim": [-10, 50]}
stime = 200
drop_first_n_steps = 25
policy_selection_steps = 25
env_reset_freq = 2

esn_tau = 5.0
esn_alpha = 0.2
esn_epsilon = 1.0e-30

arm_input = 2
arm_hidden = 100
arm_output = 3
grip_input = 44
grip_hidden = 100
grip_output = 5

internal_size = 100
visual_size = 300
somatosensory_size = 40
proprioception_size = 5
policy_size = 100 * 5
num_objects = 4

v_eradial_prop = 0.1
ss_eradial_prop = 0.1
p_eradial_prop = 0.1
a_eradial_prop = 0.1

explore_sigma = 8.0

decay = 5.0
local_decay = 1.0

base_match_sigma = 1
match_sigma = 3
base_internal_sigma = 0.5*np.sqrt(2)
internal_sigma = 30.0
base_lr = 0.005
max_lr = 1.0
stm_lr = 0.25
policy_base = np.pi*0.25
policy_noise_sigma = 0.1
policy_weights_sigma = 2
motor_noise = 1.0
representation_sigma = 5

# Modalities order: visual, touch, proprioception, action
modalities_weights = [1., 2., 1., 1.]
#modalities_weights = [0., 1., 0., 0.]  # select touch only
match_incr_th = 0.02
cum_match_stop_th = 10.0
predict_lr = 0.1
reach_grip_prop = 0.1

predict_ampl = 2
predict_base_ampl = 2
predict_ampl_prop = 0.95

epochs = 400 
batch_size = 24 
tests = 12
epochs_to_test = 100
load_weights = False
shuffle_weights = False
action_steps = 5
