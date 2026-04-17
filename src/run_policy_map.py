from params import Parameters
import SMGraphs 
g = SMGraphs.GraphManager(None, Parameters())
im = g.policy_map("weights.npy", local=True, plot=False, pca=False)
