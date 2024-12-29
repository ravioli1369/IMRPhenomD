import torch
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from phenom_d_sub import IMRPhenomD
from IMRPhenomD import gen_IMRPhenomD_hphc
from pycbc.conversions import mchirp_from_mass1_mass2, q_from_mass1_mass2

params = {
    "text.usetex": True,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": True,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "legend.title_fontsize": 16,
    "figure.titlesize": 22,
    "axes.labelpad": 10.0,
}
plt.rcParams.update(params)

"""Testing near the extremes on which the LAL implementation is reviewed"""

m1 = 10
m2 = 30
chi1 = torch.tensor([0.9])
chi2 = torch.tensor([0.9])
dist_mpc = torch.tensor([440.0])

# Frequency grid
T = 16
f_l = 16
f_sampling = 4096
# f_u = f_sampling // 2
f_u = 1024
f_ref = f_l

delta_t = 1 / f_sampling
tlen = int(round(T / delta_t))
freqs = np.fft.rfftfreq(tlen, delta_t)
df = freqs[1] - freqs[0]
fs = freqs[(freqs > f_l) & (freqs < f_u)]


m1_msun = m1
m2_msun = m2
chirp_mass = torch.tensor([mchirp_from_mass1_mass2(m1_msun, m2_msun)])
q = torch.tensor([q_from_mass1_mass2(m1_msun, m2_msun)])
tc = 0.0
phic = torch.tensor([0.0])
inclination = torch.tensor([np.pi / 2.0])
eta = q / (1 + q) ** 2

theta = [
    chirp_mass.item(),
    eta.item(),
    chi1.item(),
    chi2.item(),
    dist_mpc.item(),
    tc,
    phic.item(),
    inclination.item(),
]

fs_torch = torch.arange(f_l, f_u, df)[1:]
fs_ripple = jnp.arange(f_l, f_u, df)[1:]


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
args = parser.parse_args()

if args.torch:
    hp_torch, hc_torch = IMRPhenomD(
        fs_torch, chirp_mass, q, chi1, chi2, dist_mpc, phic, inclination, f_ref
    )
else:
    hp_jax, hc_jax = gen_IMRPhenomD_hphc(fs_ripple, theta, f_ref)
