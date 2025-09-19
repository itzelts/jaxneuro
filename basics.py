import jaxley as jx
from jaxley.channels import Na, K, Leak
import matplotlib.pyplot as plt
import numpy as np

# build the cell
comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=2)
cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

# visualize 
cell.compute_xyz() 
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = cell.vis(ax=ax, color="k")

# insert channels 
cell.insert(Leak())
cell.branch(0).insert(Na())
cell.branch(0).insert(K())

# inspect nodes 
cell.nodes
cell.branch(1).nodes # just first branch 
# explore first branche
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = cell.vis(ax=ax, color="k")
_ = cell.branch(1).vis(ax=ax, color="r")
_ = cell.branch(1).comp(1).vis(ax=ax, color="b")

# change parameters
cell.set("axial_resistivity", 200.0)

# inspect nodes again 
cell.branch(1).nodes

# visualize the morphology again 
cell.compute_xyz()
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
cell.vis(ax=ax)

cell.branch(0).set("K_gK", 0.01)  # modify potassium conductance
cell.set("v", -65.0)  # modify initial voltage

# stimulate
dt = 0.025
t_max = 10.0
current = jx.step_current(i_delay=1.0, i_dur=1.0, i_amp=0.1, delta_t=dt, t_max=t_max)

time_vec = np.arange(0, t_max+dt, dt)
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = plt.plot(time_vec, current)

# simulate one compartment 
cell.delete_stimuli() 
cell.branch(0).loc(0.0).stimulate(current)
print(cell.externals)  


# record
cell.delete_recordings() # clear first?
cell.branch(0).loc(0.0).record("v")
cell.branch(3).loc(1.0).record("v")
print(cell.recordings)  

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = cell.vis(ax=ax)
_ = cell.branch(0).loc(0.0).vis(ax=ax, color="b")
_ = cell.branch(3).loc(1.0).vis(ax=ax, color="g")

# simulate and plot
voltages = jx.integrate(cell, delta_t=dt)
print("voltages.shape", voltages.shape)
plt.plot(voltages.T)

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = ax.plot(voltages[0], c="b")
_ = ax.plot(voltages[1], c="orange")
ax.set_xlabel("step")
ax.set_ylabel("V (mV)")

plt.tight_layout()
plt.show() 