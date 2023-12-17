import numpy as np
import torch


class SMPredict:

    def __init__(self, inp_num, out_num, lr=0.1):
        self.inp_num = inp_num
        self.out_num = out_num
        self.lr = lr

        # Setting the model
        self.model = torch.nn.Linear(self.inp_num, self.out_num)
        torch.nn.init.xavier_uniform_(self.model.weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.t = 0

    def update(self, patterns, labels):
        self.optimizer.zero_grad()
        output = self.model(torch.tensor(patterns).float())
        loss = self.loss(output, torch.tensor(labels).float())
        loss.backward()
        self.optimizer.step()
        return loss

    def get_weights(self):
        # return self.out_layer.get_weights()[0]
        return self.model.weight.detach().cpu().numpy()

    def set_weights(self, weights):
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor(weights, dtype=torch.float))

    def spread(self, inp):
        assert len(inp.shape) == 2
        comp = self.model(torch.tensor(inp, dtype=torch.float)).detach().cpu().numpy()
        comp = 2*np.maximum(0, comp - 0.5)
        return comp


if __name__ == "__main__":

    inp_num = 2
    out_num = 1
    patterns_size = 10000
    epochs = 150

    labels = np.zeros(patterns_size)
    labels[patterns_size//2:] = 1
    patterns = np.vstack([labels, 1-labels]).T \
            + 0.01*np.random.randn(patterns_size, 2)
    labels = labels[:, None]
    predict = SMPredict(inp_num, out_num)

    idcs = np.arange(patterns_size)
    for t in range(epochs):
        np.random.shuffle(idcs)
        preds = predict.spread(patterns)
        loss = predict.update(patterns[idcs], labels[idcs])

        comp = 2*np.abs(preds - 0.5)
        print(loss, comp.mean())
