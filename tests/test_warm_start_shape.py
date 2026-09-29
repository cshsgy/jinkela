"""A warm-started ThermoY.forward on a new shape must not reuse the previous call's active set.

forward keeps each cell's active reactions (reaction_set, nactive) for the next warm_start call.
A caller that adjusts one shape (a column, a single cell from Python) and then warm-starts on
another (a dynamical core's block) must get the cold-start answer on the new shape, or a clean
error; not a crash, an out-of-bounds read or another answer. The sequence runs in a child
process so that a crash fails the test instead of ending the run.
"""
import os
import subprocess
import sys

import pytest
import torch
from kintera import ThermoOptions, ThermoY

torch.set_default_dtype(torch.float64)

CARD = """reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: dry, composition: {H: 1.5, He: 0.15}, cv_R: 2.5}
  - {name: NH3, composition: {N: 1, H: 3}, cv_R: 2.5, u0_R: 0.}
  - {name: H2S, composition: {H: 2, S: 1}, cv_R: 2.5, u0_R: 0.}
  - {name: NH3(s), composition: {N: 1, H: 3}, cv_R: 9.6, u0_R: -5520.}
  - {name: NH4SH(s), composition: {N: 1, H: 5, S: 1}, cv_R: 9.6, u0_R: -1.2e4}
reactions:
  - {equation: NH3 <=> NH3(s), type: nucleation, rate-constant: {formula: nh3_ideal}}
  - {equation: NH3 + H2S <=> NH4SH(s), type: nucleation, rate-constant: {formula: nh3_h2s_lewis}}
"""
# (T [K], rho [kg/m^3], mass fractions of NH3, H2S, NH3(s), NH4SH(s)),
# the first two cells of test_saturation_adjustment_nh4sh.py
CELLS = [
    (208.72174072916363, 0.8801790168247441, 5.279766354738258e-04, 1.428311049569326e-02,
     1.0099328092205351e-07, 0.),  # NH4SH 150x supersaturated, no cloud yet
    (162.72769929280207, 1.237723071684808, 3.328274229036432e-06, 8.2547022144026e-06,
     1.1644586821086675e-07, 1.2924097121314215e-03),  # NH4SH supersaturated
]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _state(th, shape, device):
    """shape cells alternating between the two CELLS; returns rho, intEng, yfrac."""
    n = 1
    for s in shape:
        n *= s
    temp, rho, *y = torch.tensor([CELLS[i % 2] for i in range(n)], device=device).t()
    temp, rho, yfrac = temp.view(shape), rho.view(shape), torch.stack(y).view((-1,) + shape)
    intEng = th.compute("VT->U", [th.compute("DY->V", [rho, yfrac]), temp])
    return rho, intEng, yfrac


def child(card, device):
    """1: warm start gave another answer than a cold start; 0: same answer, or a clean error."""
    new = (8, 8, 8)
    th = ThermoY(ThermoOptions.from_yaml(card))
    th.to(torch.device(device))
    rho, intEng, yfrac = _state(th, (1,), device)
    th.forward(rho, intEng, yfrac, False)  # leaves a one-cell active set

    rho, intEng, yfrac = _state(th, new, device)
    warm = yfrac.clone()
    try:
        th.forward(rho, intEng, warm, True)
    except RuntimeError as err:
        print("warm start on %s refused: %s" % (new, err))
        return 0

    cold = yfrac.clone()
    fresh = ThermoY(ThermoOptions.from_yaml(card))
    fresh.to(torch.device(device))
    fresh.forward(rho, intEng, cold, False)
    if not torch.equal(warm, cold):
        print("warm start on %s after %s: max |warm - cold| = %g"
              % (new, (1,), (warm - cold).abs().max().item()))
        return 1
    return 0


@pytest.mark.parametrize("device", DEVICES)
def test_warm_start_on_a_new_shape(tmp_path, device):
    (card := tmp_path / "nh4sh.yaml").write_text(CARD)
    run = subprocess.run([sys.executable, os.path.abspath(__file__), str(card), device],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, (
        "%s: rc %d (negative = killed by that signal)\n%s%s"
        % (device, run.returncode, run.stdout[-2000:], run.stderr[-2000:]))


if __name__ == "__main__":
    sys.exit(child(sys.argv[1], sys.argv[2]))
