import numpy as np
import torch
import params


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
        output = self.model(torch.tensor(patterns).float())
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
        #assert len(inp.shape) == 2
        #match = self.model(torch.tensor(inp, dtype=torch.float))
        #comp = torch.exp(-(params.match_sigma**-2) * match**2).detach().cpu().numpy()

        # OLD: Competence based on successful timesteps
        comp = self.model(torch.tensor(inp, dtype=torch.float)).detach().cpu().numpy()
        # Rescale: competence is the fraction of max n_success
        #comp = comp / params.cum_match_stop_th
        comp[comp > 1] = 1.0 # Maximum possible value is 1
        return comp


if __name__ == "__main__":

    inp_num = 2
    out_num = 1
    patterns_size = 10000
    epochs = 150

    labels = np.zeros(patterns_size)
    labels[int(patterns_size*0.25):] = 1
    labels[int(patterns_size*0.5):] = 3
    labels[int(patterns_size*0.75):] = 6
    labels[int(patterns_size*0.9):] = 8
    patterns = np.vstack([labels, 1-labels]).T \
            + 0.01*np.random.randn(patterns_size, 2)
    labels = labels[:, None]
    predict = SMPredict(inp_num, out_num)

    idcs = np.arange(patterns_size)
    for t in range(epochs):
        np.random.shuffle(idcs)
        comp = predict.spread(patterns)
        loss = predict.update(patterns[idcs], labels[idcs])
        print(loss, comp.mean())
