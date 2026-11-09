"""Fixed-capacity FIFO replay over (step, env) slots.

Slot ``i`` of env ``e`` holds the transition produced by iteration ``i``:
the state *before* the step, the action applied, and the reward/termination
that came back.  The next state lives in slot ``i+1``; when ``d[i]`` is true
that slot holds a post-reset state, so windows must not span it.

xi is constant per env (startup-only domain randomization) and ``ref_adv`` is a
pure function of (clip, frame), so neither is stored.
"""

from __future__ import annotations

import torch

from supermimic.state import NUM_JOINTS, STATE_DIM


class ReplayBuffer:
  def __init__(self, capacity: int, num_envs: int, device: str) -> None:
    self.capacity = capacity
    self.num_envs = num_envs
    self.device = device
    shape = (capacity, num_envs)
    self.s = torch.zeros(*shape, STATE_DIM, device=device)
    self.a = torch.zeros(*shape, NUM_JOINTS, device=device)
    self.r = torch.zeros(*shape, device=device)
    self.d = torch.zeros(*shape, dtype=torch.bool, device=device)
    self.trunc = torch.zeros(*shape, dtype=torch.bool, device=device)
    self.clip = torch.zeros(*shape, dtype=torch.int32, device=device)
    self.frame = torch.zeros(*shape, dtype=torch.int32, device=device)
    self.ptr = 0
    self.size = 0

  def __len__(self) -> int:
    return self.size

  @property
  def done(self) -> torch.Tensor:
    """Any episode boundary: a failure or a clip-end truncation."""
    return self.d | self.trunc

  def add(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    clip: torch.Tensor,
    frame: torch.Tensor,
    trunc: torch.Tensor | None = None,
  ) -> None:
    i = self.ptr
    self.s[i].copy_(s)
    self.a[i].copy_(a)
    self.r[i].copy_(r)
    self.d[i].copy_(d)
    self.trunc[i].copy_(trunc if trunc is not None else False)
    self.clip[i].copy_(clip)
    self.frame[i].copy_(frame)
    self.ptr = (i + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def _starts(self, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform (t, e) pairs whose next ``k`` transitions contain no reset."""
    assert self.size > k, f"replay has {self.size} steps, need > {k}"
    t = torch.randint(0, self.size - k, (n,), device=self.device)
    e = torch.randint(0, self.num_envs, (n,), device=self.device)
    return t, e

  def sample_windows(
    self, batch: int, k: int, oversample: int = 4, max_tries: int = 12
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample window starts whose states stay contiguous.

    The first ``k-1`` transitions must not end an episode, so ``s_0..s_{k-1}``
    are all logged pre-step states.  The last transition *may* be terminal:
    that is the only place a positive termination label can appear, and Eq. 1
    needs those to fit the termination head.  Its successor state is a
    post-reset state, so the caller masks that one state target.
    """
    need = batch
    ts: list[torch.Tensor] = []
    es: list[torch.Tensor] = []
    for _ in range(max_tries):
      t, e = self._starts(need * oversample, k)
      idx = t[:, None] + torch.arange(k - 1, device=self.device)[None, :]
      ok = ~self.done[idx, e[:, None]].any(dim=1)
      ts.append(t[ok])
      es.append(e[ok])
      need = batch - sum(int(x.numel()) for x in ts)
      if need <= 0:
        break
    t = torch.cat(ts)[:batch]
    e = torch.cat(es)[:batch]
    return t, e

  def sample_states(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Any valid slot, used to seed imagined rollouts."""
    assert self.size > 0
    t = torch.randint(0, self.size, (batch,), device=self.device)
    e = torch.randint(0, self.num_envs, (batch,), device=self.device)
    return t, e

  def window(self, t: torch.Tensor, e: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Gather ``k+1`` states and ``k`` actions/rewards/terminations per window."""
    idx = t[:, None] + torch.arange(k + 1, device=self.device)[None, :]
    ia = idx[:, :k]
    env = e[:, None]
    return {
      "s": self.s[idx, env.expand_as(idx)],
      "a": self.a[ia, env.expand_as(ia)],
      "r": self.r[ia, env.expand_as(ia)],
      "d": self.d[ia, env.expand_as(ia)],
      "trunc": self.trunc[ia, env.expand_as(ia)],
      "clip": self.clip[idx, env.expand_as(idx)].long(),
      "frame": self.frame[idx, env.expand_as(idx)].long(),
      "env": e,
    }

  def prev_action(self, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Action applied in slot ``t-1``; zero when slot ``t`` starts an episode."""
    has_prev = t > 0
    tp = (t - 1).clamp(min=0)
    a = self.a[tp, e]
    reset = torch.where(has_prev, self.done[tp, e], torch.ones_like(has_prev))
    return torch.where((has_prev & ~reset)[:, None], a, torch.zeros_like(a))

  def obs_history(
    self, t: torch.Tensor, e: torch.Tensor, history: int, obs_fn
  ) -> torch.Tensor:
    """Clean proprioception for the ``history`` slots ending at ``t``.

    Slots before the start of the buffer repeat the first frame, matching
    mjlab's CircularBuffer backfill.  ``obs_fn(s, a_prev, env)`` takes flattened
    (N, STATE_DIM), (N, NUM_JOINTS), (N,) tensors and returns (N, obs_dim).
    """
    offs = torch.arange(history - 1, -1, -1, device=self.device)
    idx = t[:, None] - offs[None, :]
    valid = idx >= 0
    idx_c = idx.clamp(min=0)
    env = e[:, None].expand_as(idx_c)
    flat = idx_c.reshape(-1)
    fenv = env.reshape(-1)
    s = self.s[flat, fenv]
    a_prev = self.prev_action(flat, fenv)
    o = obs_fn(s, a_prev, fenv).reshape(t.numel(), history, -1)
    return torch.where(valid[..., None], o, o[:, :1].expand_as(o))
