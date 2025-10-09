import numpy as np
import torch

from params import Parameters


params = Parameters()

class SMPredictKDE:
    """
    A simple implementation of competence predictor inspired by kernel density
    estimation. The premise is to store expected competences of each prototype
    in a grid (updated using a given learning rate). Then, for predicting
    competence of a new prototype a weighted average of competences of neighboring
    prototypes is taken using radial decay function. The radial function is taken
    directly from grid radial representation, which is the standard predictor input.
    """
    def __init__(self, grid_size, lr=0.1):
        self.competences = np.zeros(grid_size)
        self.lr = lr

    def update(self, radials, values):
        idx = radials.argmax(axis=-1)
        self.competences[idx] += self.lr*(values[:, 0] - self.competences[idx])

    def spread(self, inp):
        weights = inp / np.expand_dims(inp.sum(axis=-1), axis=-1)
        return np.expand_dims((weights * self.competences).sum(axis=-1), axis=-1)

    def get_weights(self):
        return self.competences

    def set_weights(self, weights):
        self.competences[:] = weights


class SMPredict:

    def __init__(self, inp_num, out_num, lr=0.1):
        self.inp_num = inp_num
        self.out_num = out_num
        self.lr = lr

        # Setting the model
        self.model = torch.nn.Linear(self.inp_num, self.out_num)
        torch.nn.init.xavier_uniform_(self.model.weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        self.t = 0

    def update(self, patterns, labels):
        self.optimizer.zero_grad()
        output = torch.sigmoid(self.model(torch.tensor(patterns).float()))
        loss = self.loss(output, torch.tensor(labels).float())
        loss.backward()
        self.optimizer.step()
        return loss

    def get_weights(self):
        return self.model.weight.detach().cpu().numpy()

    def set_weights(self, weights):
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor(weights, dtype=torch.float))

    def spread(self, inp):
        # assert len(inp.shape) == 2
        # match = self.model(torch.tensor(inp, dtype=torch.float))
        # comp = torch.exp(-(params.match_sigma**-2) * match**2).detach().cpu().numpy()

        # OLD: Competence based on successful timesteps
        comp = (
            torch.sigmoid(self.model(torch.tensor(inp, dtype=torch.float)))
            .detach()
            .cpu()
            .numpy()
        )
        # Rescale: competence is the fraction of max n_success
        # comp = comp / params.cum_match_stop_th
        # comp[comp > 1] = 1.0 # Maximum possible value is 1
        return comp


if __name__ == "__main__":

    grid_side = 10
    x = np.arange(grid_side)
    x = np.stack(np.meshgrid(x, x)).reshape(2, -1).T
    goal_grid = np.exp(
        -0.5
        * (2**-2)
        * np.linalg.norm(
            x.reshape(-1, 1, 2) - x.reshape(1, -1, 2),
            axis=2,
        )
        ** 2
    )
    goal_grid /= goal_grid.sum(axis=1)

    predictor = SMPredictKDE(grid_side**2, lr=1.0)

    X = np.array([goal_grid[2*10 + 3], goal_grid[0*10 + 0], goal_grid[8*10 + 7]])
    y = np.array([[1.0, 1.0, 1.0]]).T

    predictor.update(X, y)
    
    X1 = np.array([goal_grid[1*10 + 1], goal_grid[5*10 + 5], goal_grid[7*10 + 7]])

    print(predictor.spread(X))
    print(predictor.spread(X1))

    # inp_num = 2
    # out_num = 1
    # patterns_size = 10000
    # epochs = 150
    #
    # labels = np.zeros(patterns_size)
    # labels[int(patterns_size * 0.25) :] = 1
    # labels[int(patterns_size * 0.5) :] = 3
    # labels[int(patterns_size * 0.75) :] = 6
    # labels[int(patterns_size * 0.9) :] = 8
    # patterns = np.vstack([labels, 1 - labels]).T + 0.01 * np.random.randn(
    #     patterns_size, 2
    # )
    # labels = labels[:, None]
    # predict = SMPredict(inp_num, out_num)
    #
    # idcs = np.arange(patterns_size)
    # for t in range(epochs):
    #     np.random.shuffle(idcs)
    #     comp = predict.spread(patterns)
    #     loss = predict.update(patterns[idcs], labels[idcs])
    #     print(loss, comp.mean())
