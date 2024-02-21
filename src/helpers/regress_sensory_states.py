# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Add parent directory to Python module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SensoryRegressor import SensoryRegressor as Regress
# %%
# Manage data
def regress():

    # %%
    data_dict = np.load("objects_data.npy", allow_pickle=True)[0]
    n = len(data_dict)
    idcs = np.arange(n)
    np.random.shuffle(idcs)

    # %%
    inputs = np.vstack([data_dict[x]["observation"]["internal"].ravel()
        for x in range(n)])
    oinputs = np.vstack([data_dict[x]["observation"]["VISUAL_SENSORS"].ravel()
        for x in range(n)])

    # %%

    def verify_points(x, n):
        if x.T.shape[0] == n-1:
            x = np.vstack([x.T, x.T[-1]]).T
        m = np.mean(np.linalg.norm(x.reshape(2,5), axis=0))
        x = np.hstack([x.ravel(), np.mean(m)])

        return x


    targets = np.vstack([
        np.hstack([
            data_dict[x]["context"],
            verify_points(data_dict[x]["object"]["verts"], 5).ravel(),
            data_dict[x]["object"]["pos"],
            data_dict[x]["object"]["rot"],
            data_dict[x]["object"]["color"]])
        for x in range(n)])


    min_rot = targets[:, -4].min()
    max_rot = targets[:, -4].max()
    rots =  targets[:, -4]
    rots = (rots - min_rot)/(max_rot - min_rot)

    # %%

    items, input_shape = oinputs.shape
    _, output_shape = targets.shape 
    out_shapes = [1, 1, 3]

        # %%
    # build model

    layers_n = [1500,   out_shapes[0]] 
    regress0 = Regress(layers_n, input_shape, eta=0.001)
    
    layers_n = [1500, 1000,  out_shapes[1]] 
    regress1 = Regress(layers_n, input_shape, eta=0.0001)
    
    layers_n = [1500,   out_shapes[2]] 
    regress2 = Regress(layers_n, input_shape, eta=0.005)

    # # %%
    # fit

    regress0.fit(oinputs, targets[:,0], epochs= 30, batch_size=200)
    regress1.fit(oinputs, rots, epochs=500, batch_size=500)
    regress2.fit(oinputs, targets[:,-3:], epochs= 30, batch_size=200)

    # %%
    # save

    regress1.save("regress1_model")
    np.save("regress1_data", [{"rot_min":min_rot, "rot_max": max_rot}])
    regress2.save("regress2_model")
    regress0.save("regress0_model")
    print("Regress: Done!!!")

if __name__ == "__main__":
    regress()

