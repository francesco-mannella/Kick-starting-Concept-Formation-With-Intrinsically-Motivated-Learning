import box2dsim
import matplotlib.pyplot as plt
import numpy as np
import torch

from params import Parameters
from SMEnv import SMEnv
from stm import STM


_ = box2dsim
params = Parameters()


def get_data(trials, stime, env, render):
    """Collects data from the environment.

    Args:
        trials: Number of trials to collect data for.
        stime: Simulation time for each trial.
        env: The environment instance to collect data from.
        render: The rendering mode of the environment

    Returns:
        data: The collected data as a numpy array.
    """
    data = np.zeros(
        [
            trials,
            stime,
            params.visual_size
            + params.somatosensory_size
            + params.proprioception_size
            + 2,
        ]
    )
    for k in range(trials):
        chosen = np.random.choice(np.arange(1, 4))
        state = env.reset(world=chosen, render=render)
        cur_pos = state["EYE_POS"]
        arm_action = params.policy_base_arm
        grip_action = np.ones(2) * np.pi * 0.25
        action = np.hstack([arm_action, grip_action])
        print("pos epoch:", k)
        for t in range(stime):
            cur_pos = state["EYE_POS"] + np.random.randn(2) * 10
            if t % 5 == 0:
                grip_action += np.random.randn(2) * np.pi * 0.4
                grip_action = np.clip(grip_action, 0, np.pi)
            action = np.hstack([arm_action, grip_action])
            state = env.step(action)
            data[k, t] = np.hstack(
                [
                    state["VISUAL_SENSORS"].ravel(),
                    state["TOUCH_SENSORS"],
                    state["JOINT_POSITIONS"][: params.proprioception_size],
                    cur_pos,
                ]
            )
            if render is not None and render == "human":
                plt.pause(0.1)
    data = data.reshape(trials * stime, -1)
    np.save("data/StoredGripGenerateData", data)
    return data


class PrototypeGenerator:
    """Generates prototypes using Self-Organizing Maps (SOM).

    This class is responsible for training a SOM to generate prototypes from
    input data.

    Attributes:
        data: The input data for training.
        items: Number of data items.
        batch_size: Size of each training batch.
        batch_num: Number of batches.
        idcs: Indices for data shuffling.
        out_num: Number of output neurons.
        inp_num: Number of input features.
        initial_sigma: Initial neighborhood size.
        min_sigma: Minimum neighborhood size.
        lr: learning rate.
        initial_modulation: Initial loss modulation.
        epochs: Number of training epochs.
        decay_window: Window for decay of learning rate and sigma.
    """

    def __init__(
        self,
        inp_num,
        out_num,
        data,
        batch_size=50,
        min_sigma=0.7,
        lr=0.01,
        initial_modulation=2.0,
        epochs=100,
    ):
        """Initializes the prototype generator.

        Args:
            inp_num: Number of input features.
            out_num: Number of output neurons.
            data: The input data for training.
            batch_size: Size of each training batch.
            min_sigma: Minimum neighborhood size.
            initial_lr: Initial learning rate.
            epochs: Number of training epochs.
        """
        self.data = data
        self.items = data.shape[0]
        self.batch_size = batch_size
        self.batch_num = self.items // batch_size
        self.idcs = np.arange(self.items)
        self.out_num = out_num
        self.inp_num = inp_num
        self.initial_sigma = out_num / 2
        self.min_sigma = min_sigma
        self.lr = lr
        self.initial_modulation = initial_modulation
        self.epochs = epochs
        self.decay_window = epochs / 10

    def __call__(self):
        """Trains the SOM and returns the learned weights.

        Returns:
            weights: The learned weights of the SOM.
        """
        # parameters
        data = self.data
        batch_size = self.batch_size
        batch_num = self.batch_num
        idcs = self.idcs
        out_num = self.out_num
        inp_num = self.inp_num
        initial_sigma = self.initial_sigma
        min_sigma = self.min_sigma
        lr = self.lr
        epochs = self.epochs
        decay_window = self.decay_window

        # Setting the model
        som_layer = STM(inp_num, out_num, initial_sigma)
        optimizer = torch.optim.Adam(som_layer.parameters(), lr=lr)

        # training
        loss = []
        for epoch in range(epochs):
            # learning rate and sigma annealing
            curr_sigma = min_sigma + initial_sigma * np.exp(
                -epoch / decay_window
            )
            curr_modulation = self.initial_modulation * np.exp(
                -epoch / decay_window
            )

            # update learning rate and sigma in the graph
            som_layer.sigma = curr_sigma

            # iterate batches
            np.random.shuffle(idcs)
            curr_loss = []
            for batch in range(batch_num):
                batch_range = idcs[
                    np.arange(batch_size * batch, batch_size * (1 + batch))
                ]
                curr_data = torch.tensor(data[batch_range])
                optimizer.zero_grad()
                output = som_layer(curr_data)
                loss_ = curr_modulation * som_layer.loss(output)
                loss_.backward()
                optimizer.step()
                curr_loss.append(loss_.detach().numpy())
            loss.append(np.mean(curr_loss))
            print(epoch, loss[-1])

        weights = som_layer.kernel.detach().numpy()
        return weights


def generate_grip_mapping(
    inner_domain_shape, env, trials=1000, stime=50, render=None
):
    """Generates a topological mapping for grip.

    Args:
        inner_domain_shape: Shape of the inner domain for mapping.
        env: The environment instance.
        trials: Number of trials for data collection.
        stime: Simulation time for each trial.
        render: The rendering mode of the environment

    Returns:
        weights: The generated topological mapping weights.
    """
    data = get_data(trials, stime, env, render)

    visual_inp_shape = params.visual_size
    touch_inp_shape = params.somatosensory_size
    posture_inp_shape = 5
    pos_inp_shape = 2

    print("touch mapping")
    # train touch SOM and get weights
    start = visual_inp_shape
    touch = data[:, start : (start + touch_inp_shape)]

    touch = touch[touch.sum(1) > 0, :]
    touchWeights = PrototypeGenerator(
        touch_inp_shape, inner_domain_shape // 3, touch
    )()

    # train posture SOM and get weights
    print("posture mapping")
    start = visual_inp_shape + touch_inp_shape
    posture = data[:, start : (start + posture_inp_shape)]
    postureWeights = PrototypeGenerator(
        posture_inp_shape, inner_domain_shape // 3, posture
    )()

    print("pos mapping")
    pos_inp_shape = 2
    start = visual_inp_shape + touch_inp_shape + posture_inp_shape
    pos = data[:, start : (start + pos_inp_shape)]
    posWeights = PrototypeGenerator(
        pos_inp_shape, inner_domain_shape // 3, pos
    )()

    subdomain_shape = inner_domain_shape // 3
    weights = np.zeros(
        [touch_inp_shape + posture_inp_shape + 2, inner_domain_shape]
    )
    weights[:touch_inp_shape, :subdomain_shape] = touchWeights
    weights[
        touch_inp_shape : touch_inp_shape + posture_inp_shape,
        subdomain_shape : 2 * subdomain_shape,
    ] = postureWeights
    weights[touch_inp_shape + posture_inp_shape :, 2 * subdomain_shape :] = (
        posWeights
    )

    return weights


if __name__ == "__main__":
    num_hidden = 25 * 3
    params.internal_size = num_hidden
    env = SMEnv(0, params)
    weights = generate_grip_mapping(num_hidden, env, trials=1000)
    np.save("/tmp/StoredGripActuatorMap", weights)
