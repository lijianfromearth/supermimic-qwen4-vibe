"""Verify fix5 reproduces fix1's exact encoder/decoder/critic initialisation.

fix1 built a 77-output dynamics head; fix5 builds a 76-output head plus a
separate termination head in a forked RNG stream.  The padding draw in
setup.build must compensate every RNG draw the smaller head no longer consumes
(nn.Linear default init AND the orthogonal re-init), or the policy/critic init
shifts and training diverges.  This script searches the padding size empirically.
"""

from __future__ import annotations

import torch

from supermimic.rl.networks import (
  Critic,
  Dynamics,
  MLP,
  MotionEncoder,
  PolicyDecoder,
  TerminationHead,
)
from supermimic.state import NUM_JOINTS, REF_ADV_DIM, STATE_DIM, XI_DIM

GD, OD, H = 670, 93, 10
IN_DIM = STATE_DIM + NUM_JOINTS + XI_DIM + REF_ADV_DIM


def fix1_tail():
  torch.manual_seed(0)
  MLP(IN_DIM, (2048, 2048, 2048), STATE_DIM + 1, out_gain=0.1)
  return (
    MotionEncoder(GD),
    PolicyDecoder(32, OD, H),
    Critic(GD),
  )


def fix5_tail(pad: int):
  torch.manual_seed(0)
  Dynamics()
  if pad:
    torch.empty(pad).uniform_()
  enc, dec, cri = MotionEncoder(GD), PolicyDecoder(32, OD, H), Critic(GD)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(torch.initial_seed() + 1)
    TerminationHead()
  return enc, dec, cri


def eq(a, b):
  return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def main():
  ref = fix1_tail()
  for pad in (0, 2048, 4096, 4097, 4098, 4100):
    cand = fix5_tail(pad)
    flags = [eq(r, c) for r, c in zip(ref, cand)]
    print(f"pad={pad:5d} ENC={flags[0]} DEC={flags[1]} CRI={flags[2]}")
    if all(flags):
      print(f"ALIGNED_PAD={pad}")
      return
  print("NO_PAD_ALIGNED")
  raise SystemExit(1)


if __name__ == "__main__":
  main()