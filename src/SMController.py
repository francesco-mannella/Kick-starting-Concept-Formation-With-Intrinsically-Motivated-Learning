import glob
import os

import numpy as np

from params import Parameters
from SMPredict import SMPredictKDE
from stm import SMSTM


def softmax(x, lmb=1):
    e = np.exp((x - np.max(x)) / lmb)
    return e / sum(e)


def flt(n, s=0.1):
    e = np.exp(-0.5 * (s**-2) * np.linspace(-0.5, 0.5, n) ** 2)
    return e / e.sum()


class SMController:
    def __init__(self, params, sm, rng=None, load=False, shuffle=False, tag=None):

        self.sm = sm
        self.params = params
        self.maxmatch = None
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

        self.stm_v = SMSTM(
            learning_rate=self.params.stm_lr * self.params.v_lr,
            input_size=self.params.visual_size,
            output_size=self.params.internal_size,
            sigma=self.params.internal_sigma,
            external_radial_prop=self.params.v_eradial_prop,
        )
        self.stm_ss = SMSTM(
            learning_rate=self.params.stm_lr * self.params.ss_lr,
            input_size=self.params.somatosensory_size,
            output_size=self.params.internal_size,
            sigma=self.params.internal_sigma,
            external_radial_prop=self.params.ss_eradial_prop,
        )
        self.stm_p = SMSTM(
            learning_rate=self.params.stm_lr * self.params.p_lr,
            input_size=self.params.proprioception_size,
            output_size=self.params.internal_size,
            sigma=self.params.internal_sigma,
            external_radial_prop=self.params.p_eradial_prop,
        )
        self.stm_a = SMSTM(
            learning_rate=self.params.stm_lr * self.params.a_lr,
            input_size=self.params.policy_size,
            output_size=self.params.internal_size,
            sigma=self.params.internal_sigma,
            external_radial_prop=self.params.a_eradial_prop,
            weights_init_sigma=self.params.policy_weights_sigma,
        )
        if load is True:
            self.load(tag=tag, shuffle=shuffle)

        # self.predict = SMPredict(
        #      self.params.internal_size, 1, lr=self.params.predict_lr
        # )
        self.predict = SMPredictKDE(self.params.internal_size, lr=self.params.predict_lr)

        # weights_path = (
        #     pathlib.Path(__file__).parent.resolve() / "policy_weights_random.npy"
        # )
        # initial_policy = np.load(weights_path, allow_pickle=True)
        # self.stm_a.set_weights(initial_policy)

        self.match_sigma = self.params.match_sigma
        self.sigma = self.params.internal_sigma
        self.curr_sigma = self.sigma
        # self.comp_sigma = self.params.base_internal_sigma
        self.comp_sigma = self.params.representation_sigma
        self.curr_lr = None
        self.reach_cov = None
        self.reach_basis = None
        self.policy_baseline = None

        self.base_policy_noise = self.params.base_policy_noise
        self.max_policy_noise = self.params.max_policy_noise

        self.internal_side = int(np.sqrt(self.params.internal_size))
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
        representations = self.stm_a.getRepresentation(
            points, sigma=self.params.base_internal_sigma
        )
        policies = self.getPoliciesFromRepresentations(representations)
        return policies, representations

    def add_noise_to_vector_maintaining_norm(self, vector, noise_level=0.1):
        noise = self.rng.randn(*vector.shape)

        # Normalize the noise to the desired level
        norm_noise = (
            noise_level
            * np.linalg.norm(vector, axis=-1)[:, None]
            * noise
            / np.linalg.norm(noise, axis=-1)[:, None]
        )

        # Orthogonalize the noise
        noise_orthogonal = (
            norm_noise
            - np.sum(vector * norm_noise, axis=-1)[:, None]
            * vector
            / np.linalg.norm(vector, axis=-1)[:, None] ** 2
        )
        # Add orthogonal noise to the original vector
        noisy_vector = vector + noise_orthogonal
        # Rescale to maintain original norm
        noisy_vector = (
            np.linalg.norm(vector) * noisy_vector / np.linalg.norm(noisy_vector)
        )

        return noisy_vector

    def getCompetenceGrid(self):
        comp = self.predict.spread(self.goal_grid)
        # comp = 2 * (np.maximum(0.5, comp) - 0.5)
        comp = np.tanh(self.params.decay * comp)
        return comp

    def processLocalCompetence(self, comp):
        # comp = 2 * (np.maximum(0.5, comp) - 0.5)
        comp = np.tanh(self.params.local_decay * comp)
        return comp

    def get_global_local_competence(self, unprocessed_local_comp):
        self.comp_grid = self.getCompetenceGrid()
        global_comp = self.comp_grid.mean()
        local_comp = self.processLocalCompetence(unprocessed_local_comp)
        global_incompetence = 1 - global_comp
        local_incompetence = global_incompetence * (1 - local_comp)
        return {
            "global_competence": global_comp,
            "global_incompetence": global_incompetence,
            "local_competence": local_comp,
            "local_incompetence": local_incompetence,
        }

    def getPoliciesFromPointsWithNoise(self, points):
        policies, representations = self.getPoliciesFromPoints(points)
        goal_competence = self.predict.spread(representations)

        # Modulating policy exploration noise according to local competence
        comp_dict = self.get_global_local_competence(goal_competence)

        noise_sigma = (
            self.base_policy_noise
            + (self.max_policy_noise - self.base_policy_noise)
            * comp_dict["local_incompetence"]
        )
        rel = self.params.policy_noise_rel
        if rel > 0:
            rms = np.sqrt(np.mean(self.stm_a.get_weights() ** 2))
            noise_sigma = np.minimum(noise_sigma, max(self.base_policy_noise, rel * rms))

        noise = self.rng.randn(*policies.shape)
        k = self.params.policy_noise_subspace_k
        if self.params.policy_noise_reach_k > 0 and self.reach_basis is not None and len(noise) > 0:
            Q = self.reach_basis
            n_hid = Q.shape[0]
            noise = noise.reshape(len(noise), n_hid, -1)
            coef = np.einsum("hk,nho->nko", Q, noise)
            if self.params.reach_var_profile == "eig":
                coef *= self.reach_scale[None, :, None]
            noise = np.einsum("hk,nko->nho", Q, coef).reshape(len(noise), -1)
        elif k > 0:
            protos = self.stm_a.get_weights().T
            basis = np.linalg.svd(protos - protos.mean(0), full_matrices=False)[2][:k]
            sub = (noise @ basis.T) @ basis * np.sqrt(noise.shape[-1] / k)
            floor = self.params.policy_noise_floor
            noise = np.sqrt(1 - floor) * sub + np.sqrt(floor) * noise
        policies = policies + noise_sigma * noise

        return (
            policies,
            goal_competence,
            comp_dict["local_competence"],
            0 if len(noise_sigma) == 0 else noise_sigma.mean(),
        )

    def computeMatchSimple(self, v_p, ss_p, p_p, a_p, g_p, ss_contact=None):
        mods = np.stack([v_p, ss_p, p_p, a_p])
        diffs = np.moveaxis(np.linalg.norm(mods - g_p, axis=-1), 0, -1)
        match_per_mod = np.exp(-(self.match_sigma**-2) * (diffs**2))
        if self.params.touch_gate and ss_contact is not None:
            match_per_mod[..., 1] *= ss_contact
        weights = self.params.modalities_weights
        match = np.average(match_per_mod, axis=-1, weights=weights)
        if self.params.match_mode != "arithmetic":
            log_match = np.log(np.maximum(match_per_mod, 1e-300))
            geometric = np.exp(np.average(log_match, axis=-1, weights=weights))
            if self.params.match_mode == "geometric":
                match = geometric
            elif self.params.match_mode == "mixed":
                match = 0.5 * match + 0.5 * geometric
            else:
                raise ValueError(f"unknown match_mode {self.params.match_mode!r}")
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
            y = x.reshape(-1, self.params.stime)
            n, stime = y.shape
            incr = np.maximum(0, np.diff(y))
            incr = np.hstack([np.zeros([n, 1]), incr])

            return incr

        matches_increments_per_mod = np.stack(
            [
                np.stack(
                    [get_incr(matches_per_mod[:, row, col]).ravel() for col in range(5)]
                )
                for row in range(5)
            ]
        ).transpose(2, 0, 1)
        matches_increments = np.sum(matches_increments_per_mod, axis=(1, 2)) / np.sum(
            mask
        )

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

        return (
            matches,
            matches_increments.ravel(),
            matches_per_mod,
            matches_increments_per_mod,
        )

    def choose_policy(self, v_pt, v_rt, ss_rt, p_rt, goal_activation, t):

        v_rt_w = self.predict.spread(v_rt)
        v_rt_w_sum = v_rt_w.sum(axis=1)
        v_rt = (v_rt * v_rt_w).sum(axis=1) / np.where(v_rt_w_sum != 0, v_rt_w_sum, 1)
        goals_out = v_rt  # TEST: Visual modality only
        #
        goals_p, goals = self.stm_a.get_point_and_representation(
            goals_out, sigma=self.params.representation_sigma
        )

        # update policies in successful episodes
        (policies, competences, local_competences, mean_policy_noise) = (
            self.getPoliciesFromPointsWithNoise(goals_p)
        )

        return (
            goals_p,
            goals,
            policies,
            competences,
            local_competences,
            mean_policy_noise,
        )

    def spread(self, inps):

        if len(inps) == 4:
            inps.append(inps[-1])

        v, ss, p, a, g_out = inps
        v_out = self.stm_v.spread(v)
        ss_out = self.stm_ss.spread(ss)
        p_out = self.stm_p.spread(p)
        a_out = self.stm_a.spread(a)

        sigma = self.params.base_internal_sigma
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

    def update_policy_map(self, policies, modulate, goals, lr):
        mode = self.params.policy_winner
        winners = goals.argmax(1) if mode == "forced" else None
        anchor = goals if mode == "anchored" else None
        if self.params.policy_update == "ema":
            weights = goals / goals.max(1, keepdims=True) * lr
            return self.stm_a.update(
                policies, weights, winners=winners, ema_eta=self.params.policy_ema_eta, anchor=anchor
            ).item()
        return self.stm_a.update(policies, modulate, winners=winners, anchor=anchor).item()

    def update_reach_basis(self, ssensories, proprios, actuator, n_rows=300, decay=0.9):
        rows = self.rng.choice(len(ssensories), min(n_rows, len(ssensories)), replace=False)
        eye = actuator.input_mean[ssensories.shape[1] + proprios.shape[1] :]
        H = np.stack([actuator.hidden(np.hstack([ssensories[i], proprios[i], eye])) for i in rows])
        C = H.T @ H / len(H)
        self.reach_cov = C if self.reach_cov is None else decay * self.reach_cov + (1 - decay) * C
        w, v = np.linalg.eigh(self.reach_cov)
        k = self.params.policy_noise_reach_k
        self.reach_basis = v[:, -k:]
        self.reach_scale = np.sqrt(np.maximum(w[-k:], 1e-12) / np.maximum(w[-k:], 1e-12).mean())

    def baseline_select(self, sel, match_value, goals, local_lr):
        rate = self.params.policy_update_baseline
        if self.policy_baseline is None:
            self.policy_baseline = np.full(goals.shape[1], np.nan)
        idx = np.where(sel)[0]
        nodes = goals[idx].argmax(1)
        base = self.policy_baseline[nodes]
        base = np.where(np.isnan(base), match_value[idx].mean(), base)
        adv = match_value[idx] - base
        for i, n in zip(idx, nodes):
            b = self.policy_baseline[n]
            self.policy_baseline[n] = match_value[i] if np.isnan(b) else b + rate * (match_value[i] - b)
        soft = self.params.policy_baseline_soft
        if soft > 0:
            keep = np.ones(len(idx), bool)
            w = np.exp((adv - adv.max()) / soft)
            w /= w.mean()
        else:
            keep = adv > 0
            w = np.zeros(len(idx))
            if keep.any():
                w[keep] = adv[keep] / adv[keep].max()
                if self.params.policy_baseline_norm:
                    w *= len(idx) / w.sum()
        sel = np.zeros_like(sel)
        sel[idx[keep]] = True
        lr = local_lr.copy()
        if keep.any():
            lr[idx[keep]] = local_lr[idx[keep]] * w[keep].reshape((-1,) + (1,) * (local_lr.ndim - 1))
        return sel, lr

    def project_policies(self, policies, ssensories, proprios, goals, sel, actuator):
        protos = self.stm_a.get_weights().T
        n_hid, n_out = actuator.params.shape
        eye = actuator.input_mean[ssensories.shape[1] + proprios.shape[1] :]
        out = policies.copy()
        for i in np.where(sel)[0]:
            j = i
            while j > 0 and np.array_equal(policies[j - 1], policies[i]):
                j -= 1
            states = np.hstack(
                [ssensories[j : i + 1], proprios[j : i + 1], np.tile(eye, (i + 1 - j, 1))]
            )
            H = np.stack([actuator.hidden(s) for s in states])
            _, sv, vt = np.linalg.svd(H, full_matrices=False)
            Q = vt[sv > 1e-6 * sv[0]]
            proto = protos[goals[i].argmax()].reshape(n_hid, n_out)
            delta = policies[i].reshape(n_hid, n_out) - proto
            out[i] = (proto + Q.T @ (Q @ delta)).ravel()
        return out

    def update_sensory_map(self, stm, data, modulate, goals, lr):
        anchor = goals if self.params.map_winner == "anchored" else None
        if self.params.map_update == "ema":
            weights = goals / goals.max(1, keepdims=True) * lr
            return stm.update(data, weights, ema_eta=self.params.map_ema_eta, anchor=anchor).item()
        return stm.update(data, modulate, anchor=anchor).item()

    def update(
        self,
        visuals,
        ssensories,
        proprios,
        policies,
        goals,
        match_value,
        match_ind,
        max_match,
        policy_changed,
        policy_ended,
        local_lr,
        local_sigma,
        match_per_mod=None,
        actuator=None,
    ):
        curr_loss = None
        mean_modulation = None

        cgoals = goals * local_lr

        # compute number of chosen patterns (return)
        n_items = sum(match_ind)

        # TODO: Do we need hard and soft attention filtering simultaneously?
        # Furthermore, the soft part should be relative: normalized in relation
        # to maximum value.  In the supervised version match_value is binary (0
        # or 1), so there is not soft filtering effectively.  For now, we
        # simplify to hard filtering.  modulate = cgoals[match_ind] *
        # match_value[match_ind, None]
        modulate_effect = cgoals[match_ind]
        mean_modulation = modulate_effect.mean()

        local_sigma_effect = local_sigma[match_ind]

        # Condition modulation uses policy_selection_steps and competence
        # statistics of the goal selected at timestep i.
        cond_ind = np.zeros(match_ind.shape, dtype=np.bool)
        local_sigma_cond = np.zeros(local_sigma.shape)
        cgoals_cond = np.zeros(cgoals.shape)
        goals_cond = np.zeros(goals.shape)
        lr_cond = np.zeros(local_lr.shape)
        for i in np.where(policy_changed == 1)[0]:
            cond_ind[i - self.params.policy_selection_steps : i] = 1
            local_sigma_cond[i - self.params.policy_selection_steps : i] = local_sigma[
                i, None
            ]
            cgoals_cond[i - self.params.policy_selection_steps : i] = cgoals[i, None]
            goals_cond[i - self.params.policy_selection_steps : i] = goals[i, None]
            lr_cond[i - self.params.policy_selection_steps : i] = local_lr[i, None]

        local_sigma_cond = local_sigma_cond[cond_ind]
        modulate_cond = cgoals_cond[cond_ind]

        # update maps
        if n_items > 0:
            self.stm_v.update_params(sigma=local_sigma_cond)
            self.stm_ss.update_params(sigma=local_sigma_effect)
            self.stm_p.update_params(sigma=local_sigma_effect)
            self.stm_a.update_params(sigma=local_sigma_effect)
            if self.params.policy_noise_reach_k > 0 and actuator is not None:
                self.update_reach_basis(ssensories, proprios, actuator)
            policy_sel = np.asarray(match_ind, bool) & (
                match_value >= self.params.policy_update_min_match
            )
            th = self.params.policy_update_min_mod_match
            if th > 0 and match_per_mod is not None:
                policy_sel &= (match_per_mod[:, 1] >= th) & (match_per_mod[:, 2] >= th)
            lr_pol = local_lr
            if self.params.policy_update_baseline > 0 and policy_sel.any():
                policy_sel, lr_pol = self.baseline_select(policy_sel, match_value, goals, local_lr)
            if policy_sel.any():
                self.stm_a.update_params(sigma=local_sigma[policy_sel])
            policies_upd = policies
            if self.params.policy_output_space and actuator is not None and policy_sel.any():
                policies_upd = self.project_policies(
                    policies, ssensories, proprios, goals, policy_sel, actuator
                )
            curr_loss = (
                self.update_sensory_map(
                    self.stm_v, visuals[cond_ind], modulate_cond, goals_cond[cond_ind], lr_cond[cond_ind]
                ),
                self.update_sensory_map(
                    self.stm_ss, ssensories[match_ind], modulate_effect, goals[match_ind], local_lr[match_ind]
                ),
                self.update_sensory_map(
                    self.stm_p, proprios[match_ind], modulate_effect, goals[match_ind], local_lr[match_ind]
                ),
                (
                    self.update_policy_map(
                        policies_upd[policy_sel], cgoals[policy_sel], goals[policy_sel], lr_pol[policy_sel]
                    )
                    if policy_sel.any()
                    else 0.0
                ),
            )

        # Update predictor: predictor predicts cumulated matches for a
        # particular goal.
        # Calculate match distance using the match value and match sigma.
        # match_distance = np.sqrt(np.log(match_value) /
        #                          -self.params.match_sigma**-2)
        # Reshape match distance to match batch size and time steps.
        # match_distance = match_value.reshape((self.params.batch_size,
        #                                       self.params.stime, -1))

        # Predictor is updated based on the max_match achieved for each goal.
        # If no timesteps were selected for a given goal (i.e., max_match = 0),
        # this goal is not used for update.
        predictor_update_steps = policy_ended & (max_match > 0)

        self.predict.update(
            goals[predictor_update_steps],
            max_match[predictor_update_steps, None],
        )
        # self.predict.update(goals[match_ind], match_value[match_ind, None])

        return n_items, match_ind, curr_loss, mean_modulation

    def __getstate__(self):
        return {
            "params": self.params.__getstate__(),
            "visual": self.stm_v.get_weights(),
            "ssensory": self.stm_ss.get_weights(),
            "proprio": self.stm_p.get_weights(),
            "policy": self.stm_a.get_weights(),
            "predict": self.predict.get_weights(),
            "rng": self.rng,
            "maxmatch": self.maxmatch,
        }

    def __setstate__(self, state):
        params = Parameters()
        params.__setstate__(state["params"])
        self.stm_v.set_weights(state["visual"])
        self.stm_ss.set_weights(state["ssensory"])
        self.stm_p.set_weights(state["proprio"])
        self.stm_a.set_weights(state["policy"])
        self.predict.set_weights(state["predict"])
        self.rng = state["rng"]
        self.maxmatch = state["maxmatch"]

    def save(self, epoch, tag=None):

        suffix = "" if tag is None else f"-{tag}"

        weights = {
            "visual": self.stm_v.get_weights(),
            "ssensory": self.stm_ss.get_weights(),
            "proprio": self.stm_p.get_weights(),
            "policy": self.stm_a.get_weights(),
            "predict": self.predict.get_weights(),
        }

        np.save(
            f"{self.sm.epoch_dir}/weights{suffix}",
            [weights],
            allow_pickle=True,
        )

        np.save(
            f"{self.sm.site_dir}/weights{suffix}",
            [weights],
            allow_pickle=True,
        )
        np.save(self.sm.site_dir / f"comp_grid{suffix}", self.comp_grid)

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

    smcontrol = SMController()
