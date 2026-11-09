"""CPU regression test: the FIFO ring must never yield windows that span the
oldest/newest wrap boundary.

Before the fix, ``sample_windows``/``obs_history`` treated the slot index as
chronological order; once the ring wrapped, ~k/(capacity-k) of windows gathered
states ~capacity iterations apart, and their huge MSE dominated the world-model
gradient.  Run: python -m supermimic.scripts.replay_test
"""

from __future__ import annotations

import torch

from supermimic.rl.replay import ReplayBuffer
from supermimic.state import NUM_JOINTS, STATE_DIM


def fill(rb: ReplayBuffer, n: int) -> None:
  for it in range(n):
    s = torch.zeros(rb.num_envs, STATE_DIM)
    s[:, 0] = float(it)  # stamp the iteration into the root-z slot
    rb.add(
      s,
      torch.zeros(rb.num_envs, NUM_JOINTS),
      torch.zeros(rb.num_envs),
      torch.zeros(rb.num_envs, dtype=torch.bool),
      torch.zeros(rb.num_envs, dtype=torch.int32),
      torch.full((rb.num_envs,), it, dtype=torch.int32),
    )


def test_windows_contiguous(capacity: int, n: int, k: int = 8) -> None:
  rb = ReplayBuffer(capacity, 1, "cpu")
  fill(rb, n)
  seen = 0
  for _ in range(50):
    t, e = rb.sample_windows(2048, k)
    if t.numel() == 0:
      continue
    w = rb.window(t, e, k)
    stamps = w["s"][:, :, 0]
    diffs = stamps[:, 1:] - stamps[:, :-1]
    assert bool((diffs == 1).all()), f"non-contiguous window at capacity={capacity}"
    seen += stamps.shape[0]
  if capacity > 2 * k:
    assert seen > 0, "expected valid windows for a large ring"
  print(f"capacity={capacity} ptr={rb.ptr}: {seen} windows, all contiguous")


def test_history_backfill(capacity: int = 32, n: int = 80, history: int = 10) -> None:
  rb = ReplayBuffer(capacity, 1, "cpu")
  fill(rb, n)
  # Seed every slot; obs_fn echoes the state stamp so we can detect garbage.
  t = torch.arange(rb.size)
  e = torch.zeros(rb.size, dtype=torch.long)
  hist = rb.obs_history(t, e, history, lambda s, a, env: s[:, :1])
  stamps = hist[:, :, 0]
  # The newest frame (last column) must equal the seed slot's own stamp.
  own = rb.s[t, e, 0]
  assert torch.equal(stamps[:, -1], own), "current history frame mismatched"
  # No frame may come from the future relative to the seed (wrap garbage).
  assert bool((stamps <= own[:, None] + 1e-6).all()), "history leaked post-wrap state"
  print(f"history backfill capacity={capacity}: ok")


def main() -> None:
  torch.manual_seed(0)
  test_windows_contiguous(1000, 2500)
  test_windows_contiguous(64, 200)
  test_history_backfill()
  print("replay regression: PASS")


if __name__ == "__main__":
  main()