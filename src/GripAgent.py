import pathlib
import numpy as np

from Actuator import Actuator
from GripMapping import generate_grip_mapping

root_dir = pathlib.Path(__file__).parent

class GripAgent:

    def __init__(self, env, actuator_map_name=None,
                 actuator_weights_name=None, *args, **kargs):
        actuator_map, actuator_weights = None, None
        if actuator_map_name is not None:
            try:
                actuator_map = np.load((root_dir / actuator_map_name).with_suffix(".npy").resolve())
            except IOError:
                actuator_map = generate_grip_mapping(kargs["num_hidden"], env)
                np.save(actuator_map_name, actuator_map)
                print("Map Saved")
        if actuator_weights_name is not None:
            try:
                actuator_weights = np.load((root_dir / actuator_weights_name).with_suffix(".npy").resolve())
                actuator_weights = actuator_weights.reshape(
                        kargs["num_hidden"], kargs["num_outputs"])
            except IOError:
                print("Warning: {:} not found".format(actuator_weights_name))
        
        self.env = env
        self.grip = Actuator(env, actuator_map, actuator_weights, *args, **kargs)
        self.num_params = self.grip.num_hidden*self.grip.num_outputs

    def step(self, state):
        out = self.grip.step(state)
        out[:3] = out[:3] - 0.5
        out[3:] = 0.5*np.pi*(out[3:])

        return out

    def reset(self):
        self.grip.reset()

    def updatePolicy(self, params):
        self.grip.params = np.reshape(params,
            [self.grip.num_hidden, self.grip.num_outputs])


class Env:
    def __init__(self, box2d_env, **kargs):
        self.b2d_env = box2d_env
        self.b2d_env.set_taskspace(**params.task_space)
        self.render = None
        self.rng = self.b2d_env.rng
        self.reset()

    def step(self, action):
        self.b2d_env.step(action)
        if self.render is not None:
            self.b2d_env.render(self.render)
        return self.b2d_env.handPosInSpace()

    def reset(self):
        self.b2d_env.reset(world_id=self.b2d_env.worlds["noobject"])
        if self.render is not None:
            self.b2d_env.render(self.render)
        return self.b2d_env.handPosInSpace()


if __name__ == "__main__":

    # GrupActuator test
    from ArmActuator import Agent as ArmAgent, Env as ArmEnv
    from GripMapping import Env as GripEnv
    env = gym.make('Box2DSimOneArmOneEye-v0')

    arm_num_inputs = 2
    arm_num_hidden = 100
    arm_num_outputs = 3
    arm_agent = ArmAgent(ArmEnv(env), num_inputs=arm_num_inputs,
            num_hidden=arm_num_hidden, num_outputs=arm_num_outputs,
            actuator_map_name="data/StoredArmActuatorMap",
            actuator_weights_name="data/StoredArmActuatorWeights")

    grip_num_inputs = 4 + 2 + 2
    grip_num_hidden = 100
    grip_num_outputs = 5
    grip_agent = Agent(GripEnv(env), num_inputs=grip_num_inputs,
            num_hidden=grip_num_hidden, num_outputs=grip_num_outputs,
            actuator_map_name="data/StoredGripActuatorMap",
            actuator_weights_name="data/StoredGripActuatorWeights")
    grip_agent.updatePolicy(np.pi*(.1*np.ones([grip_num_hidden, grip_num_outputs])))

    for t in range(100000):
        if t%100 == 0:
            o = grip_agent.env.reset()
            arm_action = arm_agent.step([15,15]) + 0.5*np.random.randn(arm_num_outputs)

        grip_action = grip_agent.step(
                np.hstack([
                    [0, 0],
                    o["TOUCH_SENSORS"], 
                    o["JOINT_POSITIONS"][3:5]]))
        action = np.hstack([
            arm_action + grip_action[:3]*0.4, grip_action[3:]])
        o = grip_agent.env.step(action)
        env.render("human")
    input()
