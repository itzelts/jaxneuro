import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from jaxley.channels import Na, K, Leak
from jaxley.connect import fully_connect, connect 
import matplotlib.pyplot as plt
import jax.numpy as jnp


comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=4)
cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1, 2, 2])

# define network 
num_cells = 11
net = jx.Network([cell for _ in range(num_cells)])

# visualize 
net.compute_xyz()
net.rotate(180)
net.arrange_in_layers(layers=[10, 1], within_layer_offset=150, between_layer_offset=200)

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
_ = net.vis(ax=ax, detail="full")

# define synapses
pre = net.cell(range(10))
post = net.cell(10)
fully_connect(pre, post, IonotropicSynapse())

# more control
# pre = net.cell(0).branch(5).loc(1.0)
# post = net.cell(10).branch(0).loc(0.0)
# connect(pre, post, IonotropicSynapse())

# visualize again
fig, ax = plt.subplots(1, 1, figsize=(3, 6))
_ = net.vis(ax=ax, detail="full")

# inspect synaptic parameters
net.edges

# change first two synaptic parameters
net.select(edges=[0, 1]).set("IonotropicSynapse_gS", 0.1)  # nS
# for all
# net.set("IonotropicSynapse_gS", 0.0003)  # nS


# visualize the network again
net.compute_xyz()
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
net.vis(ax=ax, detail="full", layers=[10, 1])  # or `detail="point"`.

# simulate and record 
i_delay = 3.0  # ms
i_amp = 0.05  # nA
i_dur = 2.0  # ms

# Duration and step size.
dt = 0.025  # ms
t_max = 50.0  # ms
time_vec = jnp.arange(0.0, t_max + dt, dt)

net.insert(Na())
net.insert(K())
net.insert(Leak())

current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
net.delete_stimuli()
for stim_ind in range(10):
    net.cell(stim_ind).branch(0).loc(0.0).stimulate(current)

net.delete_recordings()
net.cell(10).branch(0).loc(0.0).record()
print(cell.externals)

s = jx.integrate(net, delta_t=dt)
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = ax.plot(s.T)

plt.tight_layout()
plt.show() 