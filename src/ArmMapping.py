import numpy as np
import torch
from stm import STM, radial2d as radial

def get_data(trials, stime, env):
    data = np.zeros([trials, stime, 2])
    for k in range(trials):
        env.reset()
        action = np.zeros(5)
        action[:3] = np.random.uniform(-1,1,3)
        action[:3] = 0.5*(action[:3]*np.pi*np.array([1.8, 1, 1]) - np.pi)
        print("pos epoch:",k)
        for t in range(stime):
            d = env.step(action)
            data[k, t] = d
    data = data.reshape(trials*stime, -1)
    data = data[(data[:,0]>10) & (data[:,1]>10)]
    np.save("data/StoredArmGenerateData", data)
    return data

def generate_prototypes(data, out_num):
    # parameters
    items = data.shape[0]
    batch_size = 50
    batch_num = items // batch_size
    idcs = np.arange(items)
    inp_num = 2
    initial_sigma = out_num/2
    min_sigma = 0.7
    initial_lr = 2.0
    epochs = 100
    decay_window = epochs/10
    loss = []
    
    # Setting the model
    som_layer = STM(inp_num, out_num, initial_sigma, radial_fun=radial)
    optimizer = torch.optim.Adam(som_layer.parameters(), lr=initial_lr)

    #training
    for epoch in range(epochs):
        # learning rate and sigma annealing
        curr_sigma = min_sigma + initial_sigma*np.exp(-epoch/decay_window)
        curr_rl = initial_lr*np.exp(-epoch/decay_window)

        # update learning rate and sigma
        som_layer.sigma = curr_sigma
        optimizer.param_groups[0]['lr'] = curr_rl

        # iterate batches
        np.random.shuffle(idcs)
        curr_loss = []
        for batch in range(batch_num):
            batch_range = idcs[np.arange(batch_size*batch, batch_size*(1 + batch))]
            curr_data = torch.tensor(data[batch_range])
            optimizer.zero_grad()
            output = som_layer(curr_data)
            loss_ = som_layer.loss(output) 
            loss_.backward()
            optimizer.step()
            curr_loss.append(loss_.detach().numpy())
        loss.append(np.mean(curr_loss))
        print(epoch, loss[-1])
    weights = som_layer.kernel.detach().numpy()
    return weights

def generate_arm_mapping(inner_domain_shape, env, trials=1000, stime=50):
    """ Generate a topological mapping"""
    # build dataset
    try:
        data = np.load("data/StoredArmGenerateData.npy")
    except IOError:
        data = get_data(trials, stime, env)
    # train SOM
    weights = generate_prototypes(data, inner_domain_shape)
    return weights

if __name__ == "__main__":
   
    data = np.load("data/StoredArmGenerateData.npy")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.subplot(111, aspect="equal")
    plt.xlim([5,25])
    plt.ylim([5,25])
    plt.scatter(*data.T, s=3)
    plt.show()
