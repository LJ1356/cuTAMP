# The posture prior: choosing human-like IK branches

How `cutamp/posture_prior.py` and `cutamp/posture_ref.npz` work, and the maths behind the two
tables. Implementation: `cutamp/posture_prior.py`, `cutamp/particle_initialization.py`
(`select_posture_branch`), `cutamp/scripts/bake_posture_ref.py`.

---

## 1. The problem

A 7-DOF arm needs only 6 degrees of freedom to place the gripper. The spare one means **infinitely
many joint configurations reach the same end-effector pose** — same hand, different elbow. Touch
your nose: your elbow can be high, low or out to the side and your fingertip never moves.

Humans teleoperating the arm pick one narrow family of those configurations, consistently. The
planner was picking essentially at random, which is why planned trajectories visited joint
configurations teleoperation never does.

## 2. Why it was random

Every plan waypoint starts with an IK solve. cuRobo optimises ~12 random seeds in parallel and each
converges to a *different* valid configuration — these are the **branches**. It then ranks them and
returns one:

```
rank = pose_error + null_space_error          # ik_solver.py:1099-1100
```

Every branch reaches the pose, so `pose_error` is ~0 for all of them, and `null_space_cfg.weight` is
**0.001** (`base_cfg.yml:66`) against a generic home retract. The tiebreaker is negligible, so the
winner is arbitrary — independently at every grasp, place and lift.

Measured consequence: **80–90% of the deviation from human configurations lives *between* plan
segments, not inside them.** That is exactly the part no trajopt cost can reach, because trajopt's
endpoints are already pinned by the time it runs. It is why `joint_density_weight` never moved it.

## 3. The fix in one sentence

Ask IK for `k` branches instead of 1, and keep the one that looks most like how a human holds the
arm. The two baked tables define "looks most like".

---

## 4. Table A — where humans put their joints

### Input

$$Q \in \mathbb{R}^{N \times J}, \qquad N = 987{,}631 \text{ frames}, \quad J = 7$$

Each row is one recorded frame of a human driving the arm (DROID, ~4,000 episodes). Frames are
filtered to non-idle only — $\|\dot q\|_2 > 0.05$ rad/s by central difference at 15 Hz. Without that
filter, long parked stretches dominate and the bands collapse onto wherever the arm happened to rest.

### The computation

For each ordered pair $(i,j)$, form the scalar difference across all frames and take two empirical
quantiles of that 1-D sample:

$$d_{ij}^{(n)} = Q_{n,i} - Q_{n,j}, \qquad
\texttt{lo}_{ij} = F^{-1}_{d_{ij}}(0.05), \qquad
\texttt{hi}_{ij} = F^{-1}_{d_{ij}}(0.95)$$

```python
for i in range(n_joints):
    for j in range(n_joints):
        if i != j:
            lo[i, j], hi[i, j] = np.percentile(qs[:, i] - qs[:, j], pct)   # pct = (5, 95)
```

$J^2 = 49$ independent 1-D quantile estimates. No fitting, no optimisation, no covariance.

### Why *differences* rather than each joint alone

This is where the signal is. In the planner's data joint 1 alone looked fine and joint 3 alone
looked fine — their **difference** was 4× too wide. A per-joint marginal is mathematically blind to
that, which is precisely why the earlier `joint_density` cost could not help.

### Structure

Since $d_{ji} = -d_{ij}$, the tables are antisymmetric with roles swapped:

$$\texttt{lo}_{ji} = -\texttt{hi}_{ij}, \qquad \texttt{hi}_{ji} = -\texttt{lo}_{ij}$$

Verified on the baked file: `lo.T == -hi` exactly, diagonal zero, width matrix symmetric. So there
are only $\binom{7}{2} = 21$ independent bands; the full 7×7 is stored for vectorisation, and the
penalty double-counts each pair — a constant factor of 2 that cannot change an argmin.

### Why quantiles and not mean ± kσ

These distributions are badly non-Gaussian:

| pair | skew | excess kurtosis | empirical 5–95 | Gaussian $\mu \pm 1.645\sigma$ | full min/max |
|---|---|---|---|---|---|
| q1−q3 | −0.76 | **28.9** | [−0.43, +0.56] w=1.00 | [−0.77, +0.84] w=**1.61** | [−5.07, +4.99] |
| q3−q5 | 0.09 | 6.5 | [−0.93, +1.07] w=2.00 | [−1.02, +1.07] w=2.09 | [−4.09, +5.20] |
| q1−q7 | 0.10 | 2.7 | [−1.39, +1.23] w=2.62 | [−1.36, +1.24] w=2.60 | [−4.20, +4.82] |

`q1−q3` has excess kurtosis **28.9** — sharply peaked with heavy tails. A Gaussian fit gives a band
**61% too wide** because $\sigma$ is inflated by rare excursions. Coverage check: the empirical band
holds 90.0% of frames by construction, the Gaussian band holds 95.0% — a whole extra tail admitted.
min/max would be useless: the full range of `q1−q3` is 10 radians, which permits everything.

---

## 5. Table B — which pairs are the planner's to choose

Table A says *where* humans put each pair. Table B says *which pairs the planner may move without
fighting the task*. It comes from geometry only — no human behaviour enters it.

### The null space

At configuration $q$ the Jacobian $J(q) \in \mathbb{R}^{6\times 7}$ maps joint velocity to
end-effector velocity, $\dot x = J(q)\dot q$. Six outputs, seven inputs, so generically
$\operatorname{rank}(J) = 6$ and there is a 1-D **null space**: a joint direction producing zero
end-effector motion. That is the elbow shuffle. Its orthogonal projector is

$$N(q) = I - J^{+}J$$

Verified numerically at a real configuration: $N = N^\top$, $N^2 = N$, $\operatorname{tr}(N) = 1.000
= 7 - 6$ — exactly one free dimension.

### Scoring a direction

For a unit vector $v$, $\|Nv\|^2 \in [0,1]$ is the **fraction of $v$ that is free motion**: 1 if
moving along $v$ leaves the gripper perfectly still, 0 if it only moves the gripper.

The pair direction is "joint $i$ up, joint $j$ down, equally", normalised so all 21 pairs are scored
on the same footing:

$$v_{ij} = \frac{e_i - e_j}{\sqrt{2}}, \qquad
w_{ij} = \mathbb{E}_q\!\left[\|N(q)\,v_{ij}\|^2\right]$$

$N(q)$ depends on $q$ — the null space rotates as the arm moves — so the expectation is over 4,000
sampled human configurations.

### The shortcut the code uses

Because $N$ is a **symmetric idempotent** projector,

$$\|Nv\|^2 = v^\top N^\top N v = v^\top N^2 v = v^\top N v$$

(verified: both give `0.059292` at the test configuration). That linearises the expectation:

$$w_{ij} = \mathbb{E}_q\!\left[v_{ij}^\top N(q) v_{ij}\right]
        = v_{ij}^\top \underbrace{\mathbb{E}_q[N(q)]}_{A} v_{ij}$$

So accumulate **one** 7×7 matrix over 4,000 configurations, then read off all 49 entries as cheap
quadratic forms instead of 49 × 4,000 projections:

```python
A = mean over q of (I - pinv(J) @ J)      # one 7x7
w[i, j] = v @ A @ v                        # v = (e_i - e_j)/sqrt(2)
```

### Does $w$ mean what it claims?

Direct check — how much end-effector motion each pair direction actually causes:

| pair | $w_{ij}$ | EE linear | EE angular |
|---|---|---|---|
| q1−q3 | **0.728** | **3.3 cm/rad** | 17 °/rad |
| q3−q5 | 0.395 | 36.1 cm/rad | 69 °/rad |
| q1−q7 | 0.263 | 40.7 cm/rad | 80 °/rad |
| q5−q7 | 0.234 | 6.5 cm/rad | 31 °/rad |
| q1−q2 | 0.177 | 59.5 cm/rad | 57 °/rad |
| q4−q6 | 0.006 | 28.1 cm/rad | 7 °/rad |
| q2−q4 | **0.002** | **80.8 cm/rad** | 81 °/rad |

High $w$ ↔ the gripper barely moves. Low $w$ ↔ the gripper flies. **This is why the penalty cannot
fight the task**: a pair that would drag the gripper off target is automatically down-weighted to
nothing. No hand-tuning, no exclusion list.

### Structural properties

- $w_{ij} \in [0,1]$, guaranteed by $N$ being a projector. Observed range `[0.002, 0.728]`.
- Symmetric, $w_{ij} = w_{ji}$: $v_{ji} = -v_{ij}$ and the form is quadratic.
- **No sum rule.** The 21 pair directions are not orthogonal, so $\sum_{i<j} w_{ij} = 3.485$, not 1.
  (An orthonormal basis would sum to $\operatorname{tr}(A) = 1$.) These are relative importances,
  not a probability distribution.

---

## 6. What happens at plan time

`select_posture_branch` runs once per IK solve, for each of the `k` returned branches:

**Step 1 — verify it reaches the target.** cuRobo's `IKResult.success` comes back **all-False** in
this batched path (and `position_error` all-zero), so branches are validated by forward kinematics
instead: within `posture_pos_tol` (5 mm) and `posture_rot_tol` (0.05 rad ≈ 2.9°). This incidentally
filters non-reaching solutions the stock path accepts blindly — max FK error over 512 endpoints was
6.19 mm / 60.7° stock versus 0.01 mm selected.

**Step 2 — score the posture.**

$$\text{penalty}(q) = \sum_{i<j} w_{ij}\Big[\max(0,\; d_{ij} - \texttt{hi}_{ij}) +
\max(0,\; \texttt{lo}_{ij} - d_{ij})\Big], \qquad d_{ij} = q_i - q_j$$

A **one-sided hinge**: exactly zero inside the human band, and the distance outside it (in radians)
otherwise, weighted by that pair's freeness.

**Step 3 — keep the argmin**, with invalid branches masked to $+\infty$.

Because in-band costs exactly zero, this does not force the arm into one posture. It pushes back
only when a branch leaves the range humans use; among acceptable branches all penalties tie at 0 and
`argmin` returns the lowest index — cuRobo's own preference.

### Fallbacks

Selection can never lose a plan the stock configuration would have found:

| situation | behaviour |
|---|---|
| no FK-valid branch for a particle | keep cuRobo's top seed (`solution[:, 0]`) |
| prior file missing or malformed | log once, disable selection |
| prior covers more joints than the arm has DOF | log once, disable selection |
| `posture_selection_seeds: 0` | returns a tensor **byte-identical** to the previous code path |

---

## 7. Configuration

One knob:

```yaml
tamp_overrides:
  posture_selection_seeds: 12     # branches to score; 0 or 1 = off
```

Everything else — which joints matter, how far they may vary — comes from the baked file. Nothing to
tune per scene, per task or per object.

`TAMPConfiguration` also exposes `posture_ref` (path to an alternative baked file) and the two FK
tolerances, all optional.

Note `ik_num_seeds` is raised to 3× the branch count automatically: cuRobo's `return_seeds` cannot
exceed the solver's `num_seeds`, and branch diversity thins as the two converge.

---

## 8. Baking and porting

```bash
# default: DROID proprio shard cache, FR3
python -m cutamp.scripts.bake_posture_ref

# a different corpus and robot
python -m cutamp.scripts.bake_posture_ref --source lerobot \
    --path ~/.cache/huggingface/lerobot/<user>/<repo> --robot fr3_robotiq
```

The script is self-contained — corpus loading, Jacobian and table build all live in it, so
regenerating the prior needs only cuTAMP, numpy and the corpus on disk.

The Jacobian is **central-differenced through cuRobo's own FK** rather than a hardcoded DH table
(cuRobo refuses to emit an analytic one: *"Outputting jacobian is not supported"*). That keeps the
baker robot-agnostic. Cross-checked against an analytic FR3 DH Jacobian: bands bit-identical, weights
max abs diff **2.6e-04**, pair ranking unchanged.

Re-bake against a different robot or corpus and the whole prior moves with it. Which joints matter is
an *output*, not an input.

---

## 9. Measured results

512 endpoints, `k = 12`, scored by a 24-component GMM fit to DROID configurations — an independent
yardstick, not the selector's own objective:

| | GMM logL | q1−q3 sd | max FK error |
|---|---|---|---|
| stock (cuRobo top seed) | −12.55 | 0.98 | 0.01 mm |
| **posture selection** | **−10.96** | 0.44 | 0.02 mm |
| DROID itself (ceiling) | −6.62 | 0.50 | — |

Cost is ~24 ms per IK batch at 512 particles, against multi-second plans.

**Nothing is hand-picked.** The baked table independently recovers `q1−q3` as by far the freest pair
(w = 0.728, band [−0.44, +0.56]), then `q3−q5` (0.395) and `q1−q7` (0.263) — the coordinate that
diagnosis had identified by hand falls out of the geometry instead of being assumed.

---

## 10. Limitations

**Table A is a product of marginals.** Each pair's band is estimated independently, so it cannot
express "`q1−q3` may be wide *when* `q5−q7` is narrow". It is the same class of simplification that
made `joint_density` fail — but one level up, on pair differences rather than single joints, which
is where the signal turned out to live.

**5/95 is a policy choice, not a fact.** It rejects the outer 10% of human behaviour. Tighter would
reject postures humans genuinely use; looser would admit the tails being removed. It is the one free
parameter in Table A.

**$\mathbb{E}[N]$ averages projectors**, which is meaningful only if the null space is reasonably
stable across the sampled region. It is here — the arm works in a tabletop volume — but on a robot
ranging over a much wider workspace the average smears and every $w$ drifts toward an uninformative
middle. The tell is the eigenvalue spread of $A$: on DROID configurations it is `[0.78, 0.20, 0.01,
…]`, one dominant direction; on uniformly sampled configurations it is `[0.43, 0.24, 0.13, …]`, and
a fixed table would be the wrong model.

**$w$ measures freeness, not brokenness.** It says the planner may move a pair without cost to the
task — not that the planner currently gets it wrong. Those correlate only **+0.40**. Table B alone
would spend effort on pairs that are already fine; pairing it with Table A's bands is what makes the
penalty zero when nothing is wrong.

**The q7 π-flip is not addressed.** 30–46% of planned frames sit on the wrist branch teleoperation
never uses. That is a property of the **grasp**, not the IK branch — the two representatives are two
different EE poses. Solving IK for both a grasp and its π-rolled twin takes q7 out-of-band from 50%
to 0%, but it must also flip the stored `obj_from_grasp` to match, so it is a separate change.
(Whoever does it: issue two same-size `solve_batch` calls — doubling the batch trips cuRobo's cuda
graph with *"changing goal type"*.)

**Everything above is offline.** Real recorded poses, real IK, real trajopt — but never a live scene
with obstacles. `k` branches are scored before collision pruning in these measurements; in clutter
some branches are pruned, so branch diversity should be re-checked on a saved run's world.
