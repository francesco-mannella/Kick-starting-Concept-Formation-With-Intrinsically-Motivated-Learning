import os
import glob
import params
import numpy as np
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
        )
        if load is True:
            self.load(tag=tag, shuffle=shuffle)

        self.predict = SMPredict(
            params.internal_size,
            1,
            lr=params.predict_lr
        )

        self.match_sigma = params.match_sigma
        self.sigma = params.internal_sigma
        self.curr_sigma = self.sigma
        self.comp_sigma = params.base_internal_sigma
        self.explore_sigma = params.explore_sigma
        self.policy_noise = None

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
        self.getCompetenceGrid()

        self.episode_mask = np.arange(params.stime*params.batch_size) % params.stime
        self.episode_mask = self.episode_mask > (params.stime*0.1)

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
        representations = self.stm_a.getRepresentation(points)
        policies = self.getPoliciesFromRepresentations(representations)
        return policies, representations

    def getPoliciesFromRepresentationsWithNoise(self, representations):
        policies = self.getPoliciesFromRepresentations(representations)
        rcomp = self.predict.spread(representations)
        comp = SMController.comp_fun(rcomp)
        self.policy_noise = self.rng.randn(*policies.shape)

        policies = self.explore_sigma * (policies + (1 - comp) * self.policy_noise)
        return policies, comp, rcomp

    def computeMatch(self, representations, target):

        repall = np.stack(representations)
        repall = np.vstack([repall, np.reshape(target, (1, -1, 2))])
        d1 = np.expand_dims(repall, 0)
        d2 = np.expand_dims(repall, 1)
        diffs = np.linalg.norm(d1 - d2, axis=-1)
        matches_all = np.exp(-0.5 * (self.match_sigma**-2) * (diffs**2))

        # take into account only distances with goal
        mask = [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
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

        return matches, matches_increments.ravel()

    def getCompetenceGrid(self):
        comp = self.predict.spread(self.goal_grid)
        return SMController.comp_fun(comp)

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
        matches,
        matches_increment,
        competences,
        pretest=False,
    ):

        if not pretest:
            cgoals = goals * (1 - competences)

            # select base on match_value
            mch_idcs = matches.ravel() > params.match_th

            # select base on match_increment
            mch_idcs &= matches_increment.ravel() > params.match_incr_th

            # mask
            mch_idcs &= self.episode_mask

            # compute number of chosen patterns (return)
            batch = len(matches)
            n_items = sum(mch_idcs)
            idcs = mch_idcs

            modulate = cgoals[idcs] * matches[idcs]

            # update maps
            if n_items > 0:

                self.stm_v.update(
                    visuals[idcs],
                    modulate,
                )
                self.stm_ss.update(
                    ssensories[idcs],
                    modulate,
                )
                self.stm_p.update(
                    proprios[idcs],
                    modulate,
                )
                self.stm_a.update(
                    policies[idcs],
                    modulate,
                )

            # find max match_value within each episode
            stime = params.stime
            m = matches.reshape(-1, stime)
            m = np.eye(stime)[np.argmax(m, 1)]
            m = m.reshape(-1)
            idcs = np.where(m == 1)

            # update predictor
            cmm = matches[idcs].max()
            self.maxmatch = cmm \
                    if self.maxmatch is None \
                    else self.maxmatch if cmm < self.maxmatch \
                    else cmm

            th = 0.2*competences[idcs]
            self.predict.update(goals[idcs], matches[idcs] > th)

        elif pretest:
            if not hasattr(self, "count"):
                self.count = 0
                self.init_data = {
                    "visuals": [],
                    "ssensories": [],
                    "proprios": [],
                    "policies": [],
                }

            mch_idcs = np.ones_like(matches.ravel()) > 0
            n_items = sum(1 * mch_idcs)

            if self.count < params.pretest_epochs:

                if n_items > 0:
                    self.init_data["visuals"].append(visuals[mch_idcs])
                    self.init_data["ssensories"].append(ssensories[mch_idcs])
                    self.init_data["proprios"].append(proprios[mch_idcs])
                    self.init_data["policies"].append(policies[mch_idcs])

            if self.count == params.pretest_epochs - 1:

                for k in self.init_data.keys():
                    self.init_data[k] = np.vstack(self.init_data[k])

                for t in range(400):
                    h = 3.0 * np.exp(-t / 15)
                    s = 0.7 + 8 * np.exp(-t / 15)
                    self.updateParams(s, h)

                    ones = np.ones(self.init_data["visuals"].shape[0])
                    lv = self.stm_v.update(self.init_data["visuals"], ones)
                    ones = np.ones(self.init_data["ssensories"].shape[0])
                    ls = self.stm_ss.update(self.init_data["ssensories"], ones)
                    ones = np.ones(self.init_data["proprios"].shape[0])
                    lp = self.stm_p.update(self.init_data["proprios"], ones)
                    ones = np.ones(self.init_data["policies"].shape[0])
                    la = self.stm_a.update(self.init_data["policies"], ones)
                    print(lv, ls, lp, la)

            self.count += 1

        return n_items, mch_idcs

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
