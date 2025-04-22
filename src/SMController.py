import os
import glob
import params
import numpy as np
import pathlib


from stm import SMSTM
from SMPredict import SMPredict


def softmax(x, lmb=1):
    e = np.exp((x - np.max(x)) / lmb)
    return e / sum(e)


def flt(n, s=0.1):
    e = np.exp(-0.5 * (s**-2) * np.linspace(-0.5, 0.5, n) ** 2)
    return e / e.sum()


class SMController:
    def __init__(self, rng=None, load=False, shuffle=False, tag=None):
        self.maxmatch = None
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

        self.stm_v = SMSTM(
            learning_rate=params.stm_lr,
            input_size=params.visual_size,
            output_size=params.internal_size,
            sigma=params.internal_sigma,
            external_radial_prop=params.v_eradial_prop,
        )
        self.stm_ss = SMSTM(
            learning_rate=params.stm_lr,
            input_size=params.somatosensory_size,
            output_size=params.internal_size,
            sigma=params.internal_sigma,
            external_radial_prop=params.ss_eradial_prop,
        )
        self.stm_p = SMSTM(
            learning_rate=params.stm_lr,
            input_size=params.proprioception_size,
            output_size=params.internal_size,
            sigma=params.internal_sigma,
            external_radial_prop=params.p_eradial_prop,
        )
        self.stm_a = SMSTM(
            learning_rate=params.stm_lr,
            input_size=params.policy_size,
            output_size=params.internal_size,
            sigma=params.internal_sigma,
            external_radial_prop=params.a_eradial_prop,
            weights_init_sigma=params.policy_weights_sigma
        )
        if load is True:
            self.load(tag=tag, shuffle=shuffle)

        self.predict = SMPredict(
            params.internal_size,
            1,
            lr=params.predict_lr
        )

        weights_path = pathlib.Path(__file__).parent.resolve() / "policy_weights_random.npy"
        initial_policy = np.load(weights_path, allow_pickle=True)
        #self.stm_a.set_weights(initial_policy)

        self.match_sigma = params.match_sigma
        self.sigma = params.internal_sigma
        self.curr_sigma = self.sigma
        self.comp_sigma = params.base_internal_sigma
        self.curr_lr = None

        self.base_policy_noise = params.base_policy_noise
        self.max_policy_noise = params.max_policy_noise

        self.internal_side = int(np.sqrt(params.internal_size))
        x = np.arange(self.internal_side)
        self.radial_grid = np.stack(np.meshgrid(x, x)).reshape(2, -1).T
        x = self.radial_grid
        self.goal_grid = self.goals_real = np.exp(
            -0.5
            * (self.comp_sigma**-2)
            * np.linalg.norm(
                x.reshape(-1, 1, 2) - x.reshape(1, -1, 2),
                axis=2,
            )
            ** 2
        )
        self.goal_grid /= self.goal_grid.sum(axis=1)
        self.comp_grid = self.getCompetenceGrid()

        # This effectively ensures that first 10% of simulation steps
        # of each episode is not taken into account when updating
        # sensorimotor maps based on match values.
        self.episode_mask = np.arange(params.stime*params.batch_size) % params.stime
        self.episode_mask = self.episode_mask > params.drop_first_n_steps

    @staticmethod
    def comp_fun(comp):
        basecomp = np.tanh(params.predict_base_ampl*comp)
        hypercomp = np.tanh(params.predict_ampl*comp)
        prob = params.predict_ampl_prop
        return (1 - prob) * basecomp + prob * hypercomp

    def update_reentrant_connections(self):

        a_radials = self.stm_a.get_radials()
        ss_radials = self.stm_ss.get_radials()
        p_radials = self.stm_p.get_radials()
        v_radials = self.stm_v.get_radials()

        radials = (a_radials + ss_radials + p_radials + v_radials) / 4

        self.stm_ss.set_radials(radials)
        self.stm_p.set_radials(radials)
        self.stm_a.set_radials(radials)
        self.stm_v.set_radials(radials)

    def getPoliciesFromRepresentations(self, representations):
        return self.stm_a.backward(representations)

    def getVisualsFromRepresentations(self, representations):
        return self.stm_v.backward(representations)

    def getTouchesFromRepresentations(self, representations):
        return self.stm_ss.backward(representations)

    def getPropriosFromRepresentations(self, representations):
        return self.stm_p.backward(representations)

    def getPoliciesFromPoints(self, points):
        # Setting minimal sigma guarantees that exact prototypes are returned,
        # not interpolated policies over larger portion of the grid. 
        representations = self.stm_a.getRepresentation(points, sigma=params.base_internal_sigma)
        policies = self.getPoliciesFromRepresentations(representations)
        return policies, representations

    def add_noise_to_vector_maintaining_norm(self, vector, noise_level=0.1):
        noise = self.rng.randn(*vector.shape)

        # Normalize the noise to the desired level
        norm_noise = noise_level * np.linalg.norm(vector, axis=-1)[:, None] * noise / np.linalg.norm(noise, axis=-1)[:, None]

        # Orthogonalize the noise
        noise_orthogonal = norm_noise - np.sum(vector * norm_noise, axis=-1)[:, None] * vector / np.linalg.norm(vector, axis=-1)[:, None]**2
        # Add orthogonal noise to the original vector
        noisy_vector = vector + noise_orthogonal
        # Rescale to maintain original norm
        noisy_vector = np.linalg.norm(vector) * noisy_vector / np.linalg.norm(noisy_vector)

        return noisy_vector

    def getPoliciesFromPointsWithNoise(self, points):
        policies, representations = self.getPoliciesFromPoints(points)
        rcomp = self.predict.spread(representations)
        #comp = SMController.comp_fun(rcomp)
        comp = rcomp
       
        # Modulating policy exploration noise according to local competence
        global_comp = self.comp_grid.mean()
        global_incompetence = 1 - np.tanh(params.decay * global_comp)
        local_incompetence = global_incompetence * (1 - np.tanh(params.local_decay * comp))
        noise_sigma = (self.base_policy_noise + (self.max_policy_noise
            - self.base_policy_noise) * local_incompetence)

        noise = self.rng.randn(*policies.shape)
        #policies = policies + params.policy_noise_sigma*(1-comp)*noise
        policies = policies + noise_sigma*noise
        #policies = self.add_noise_to_vector_maintaining_norm(policies,
        #                    noise_level=params.policy_noise_sigma*comp)
        
        return policies, comp, rcomp, noise_sigma.mean()


    def getPoliciesFromRepresentationsWithNoise(self, representations):
        policies = self.getPoliciesFromRepresentations(representations)
        rcomp = self.predict.spread(representations)
        #comp = SMController.comp_fun(rcomp)
        comp = rcomp
       
        # Modulating policy exploration noise according to local competence
        global_comp = self.comp_grid.mean()
        global_incompetence = 1 - np.tanh(params.decay * global_comp)
        local_incompetence = global_incompetence * (1 - np.tanh(params.local_decay * comp))
        noise_sigma = (self.base_policy_noise + (self.max_policy_noise
            - self.base_policy_noise) * local_incompetence)

        noise = self.rng.randn(*policies.shape)
        #policies = policies + params.policy_noise_sigma*(1-comp)*noise
        policies = policies + noise_sigma*noise
        #policies = self.add_noise_to_vector_maintaining_norm(policies,
        #                    noise_level=params.policy_noise_sigma*comp)
        
        return policies, comp, rcomp, noise_sigma.mean()

    def computeMatchSimple(self, v_p, ss_p, p_p, a_p, g_p):
        mods = np.stack([v_p, ss_p, p_p, a_p])
        diffs = np.moveaxis(np.linalg.norm(mods - g_p, axis=-1), 0, -1)
        match_per_mod = np.exp(-(self.match_sigma**-2) * (diffs**2))
        match = np.average(match_per_mod, axis=-1, weights=params.modalities_weights)
        return match, match_per_mod

    # TODO: This method is outdated and is kept for reference
    def computeMatch(self, representations, target):
        repall = np.stack(representations)
        repall = np.vstack([repall, np.reshape(target, (1, -1, 2))])
        d1 = np.expand_dims(repall, 0)
        d2 = np.expand_dims(repall, 1)
        diffs = np.linalg.norm(d1 - d2, axis=-1)
        matches_all = np.exp(-0.5 * (self.match_sigma**-2) * (diffs**2))

        # take into account only distances with goal
        # Mod order: visual, touch, proprioception, action, goal
        mask = [
            [0, 0, 0, 0, 1.0],
            [0, 0, 0, 0, 1.0],
            [0, 0, 0, 0, 1.0],
            [0, 0, 0, 0, 1.0],
            [0, 0, 0, 0, 0],
        ]
        matches_per_mod = matches_all.transpose(2, 0, 1) * mask
        matches = np.sum(matches_per_mod, axis=(1, 2)) / np.sum(mask)

        # same for increments
        def get_incr(x):
            y = x.reshape(-1, params.stime)
            n, stime = y.shape
            incr = np.maximum(0, np.diff(y))
            incr = np.hstack([np.zeros([n, 1]), incr])

            return incr

        matches_increments_per_mod = np.stack([
            np.stack([get_incr(matches_per_mod[:, row, col]).ravel()
                for col in range(5)]) for row in range(5)]).transpose(2, 0, 1)
        matches_increments = np.sum(matches_increments_per_mod, axis=(1, 2)) / np.sum(mask)

        # # requires that all sensory modalities change
        # mask_req = [
        #     [0, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0],
        # ]
        # 
        # matches_mins = matches_increments_per_mod > 0.01
        # matches_min_for_all = np.sum(matches_mins, axis=(1, 2)) == 3 
        # matches_increments = matches_increments * matches_min_for_all

        return matches, matches_increments.ravel(), matches_per_mod, matches_increments_per_mod

    def choose_policy(self, v_rt, ss_rt, p_rt, t):
        # TODO: ugly hack to avoid division by 0
        #v_rt_w = 1.1 - self.controller.predict.spread(v_rt)
        #ss_rt_w = 1.1 - self.controller.predict.spread(ss_rt)
        #p_rt_w = 1.1 - self.controller.predict.spread(p_rt)
        v_rt_w = self.predict.spread(v_rt)
        ss_rt_w = self.predict.spread(ss_rt)
        p_rt_w = self.predict.spread(p_rt)

        v_rt = (v_rt * v_rt_w).sum(axis=1) / v_rt_w.sum(axis=1)
        ss_rt = (ss_rt * ss_rt_w).sum(axis=1) / ss_rt_w.sum(axis=1)
        p_rt = (p_rt * p_rt_w).sum(axis=1) / p_rt_w.sum(axis=1)

        #goals = np.average([v_rt, ss_rt, p_rt],
        #                   axis=0,
        #                   weights=[params.modalities_weights[0],
        #                            params.modalities_weights[1],
        #                            params.modalities_weights[2]])
        #goals = (v_rt + ss_rt + p_rt) / 3 # TEST
        #goals_out = (v_rt + p_rt) / 2 # TEST: no touch modality
        goals_out = v_rt # TEST: Visual modality only

        goals_p, goals = self.stm_a.get_point_and_representation(goals_out, sigma=params.representation_sigma) 

        # update policies in successful episodes
        (policies,
         competences,
         rcompetences,
         mean_policy_noise) = self.getPoliciesFromPointsWithNoise(goals_p)

        return goals_p, goals, policies, competences, rcompetences, mean_policy_noise

    def getCompetenceGrid(self):
        comp = self.predict.spread(self.goal_grid)
        return comp
        #return SMController.comp_fun(comp)

    def spread(self, inps):

        if len(inps) == 4:
            inps.append(inps[-1])

        v, ss, p, a, g_out = inps
        v_out = self.stm_v.spread(v)
        ss_out = self.stm_ss.spread(ss)
        p_out = self.stm_p.spread(p)
        a_out = self.stm_a.spread(a)

        sigma = params.base_internal_sigma
        v_p, v_r = self.stm_v.get_point_and_representation(v_out, sigma)
        ss_p, ss_r = self.stm_ss.get_point_and_representation(ss_out, sigma)
        p_p, p_r = self.stm_p.get_point_and_representation(p_out, sigma)
        a_p, a_r = self.stm_a.get_point_and_representation(a_out, sigma)
        g_p, _ = self.stm_a.get_point_and_representation(g_out, sigma)

        return (v_r, ss_r, p_r, a_r, g_out), (v_p, ss_p, p_p, a_p, g_p)

    def updateParams(self, sigma, lr):
        self.stm_v.update_params(sigma=sigma, lr=lr)
        self.stm_ss.update_params(sigma=sigma, lr=lr)
        self.stm_p.update_params(sigma=sigma, lr=lr)
        self.stm_a.update_params(sigma=sigma, lr=lr)
        self.sigma = sigma

    def update(
        self,
        visuals,
        ssensories,
        proprios,
        policies,
        goals,
        match_value,
        match_ind,
        cum_match,
        policy_changed,
        local_lr,
        local_sigma
    ):
        curr_loss = None
        mean_modulation = None

        cgoals = goals * local_lr

        # compute number of chosen patterns (return)
        n_items = sum(match_ind)

        # TODO: Do we need hard and soft attention filtering simultaneously?
        # Furthermore, the soft part should be relative: normalized in relation to maximum value.
        # In the supervised version match_value is binary (0 or 1), so there is not soft filtering effectively.
        # For now, we simplify to hard filtering.
        #modulate = cgoals[match_ind] * match_value[match_ind, None]
        modulate = cgoals[match_ind]
        mean_modulation = modulate.mean()

        local_sigma = local_sigma[match_ind]

        # update maps
        if n_items > 0:
            self.stm_v.update_params(sigma = local_sigma)
            self.stm_ss.update_params(sigma = local_sigma)
            self.stm_p.update_params(sigma = local_sigma)
            self.stm_a.update_params(sigma = local_sigma)
            curr_loss = (self.stm_v.update(visuals[match_ind], modulate).item(),
                         self.stm_ss.update(ssensories[match_ind], modulate).item(),
                         self.stm_p.update(proprios[match_ind], modulate).item(),
                         self.stm_a.update(policies[match_ind], modulate).item())

        # update predictor: predictor predicts cumulated matches for a particular goal
        #goals = goals.reshape((params.batch_size, params.stime, -1))
        #match_distance = np.sqrt(np.log(match_value)/-params.match_sigma**-2)
        #match_distance = match_value.reshape((params.batch_size, params.stime, -1))

        #self.predict.update(goals[policy_changed], cum_match[policy_changed, None])
        self.predict.update(goals[match_ind], match_value[match_ind, None])

        return n_items, match_ind, curr_loss, mean_modulation

    def __getstate__(self):
        return {
            "visual": self.stm_v.get_weights(),
            "ssensory": self.stm_ss.get_weights(),
            "proprio": self.stm_p.get_weights(),
            "policy": self.stm_a.get_weights(),
            "predict": self.predict.get_weights(),
            "rng": self.rng,
            "maxmatch": self.maxmatch,
        }

    def __setstate__(self, state):
        self.__init__()
        self.stm_v.set_weights(state["visual"])
        self.stm_ss.set_weights(state["ssensory"])
        self.stm_p.set_weights(state["proprio"])
        self.stm_a.set_weights(state["policy"])
        self.predict.set_weights(state["predict"])
        self.rng = state["rng"]
        self.maxmatch = state["maxmatch"]

    def save(self, epoch, tag=None):

        storage_dir = f"storage{'-' if tag is not None else '' }{tag if tag is not None else ''}"
        epoch_dir = f"{storage_dir}/{epoch:06d}"
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(epoch_dir, exist_ok=True)

        weights = {
            "visual": self.stm_v.get_weights(),
            "ssensory": self.stm_ss.get_weights(),
            "proprio": self.stm_p.get_weights(),
            "policy": self.stm_a.get_weights(),
            "predict": self.predict.get_weights(),
        }

        np.save(
            f"{epoch_dir}/weigths",
            [weights],
            allow_pickle=True,
        )

        np.save("www/visual_weights", self.stm_v.get_weights())
        np.save("www/comp_grid", self.comp_grid)

    def load(
        self,
        weights=None,
        epoch=None,
        shuffle=False,
        tag=None,
    ):

        if weights is None:
            storage_dir = f"storage{'-' if tag is not None else '' }{tag if tag is not None else ''}"
            if os.path.isdir(storage_dir):
                if epoch is None:
                    epochs = sorted(glob.glob(f"{storage_dir}/*"))
                    epoch_dir = f"{storage_dir}/{epochs[-1]}"
                else:
                    epoch_dir = f"{storage_dir}/{epoch:06d}"

                weights = np.load(
                    f"{epoch_dir}/weights.npy",
                    allow_pickle=True,
                )[0]
            else:
                raise Exception(f"{storage_dir} does not exist!")

        if shuffle == True:
            for k in weights.keys():
                self.rng.shuffle(weights[k])

        self.stm_v.set_weights(weights["visual"])
        self.stm_ss.set_weights(weights["ssensory"])
        self.stm_p.set_weights(weights["proprio"])
        self.stm_a.set_weights(weights["policy"])
        self.predict.set_weights(weights["predict"])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    smcontrol = SMController()
