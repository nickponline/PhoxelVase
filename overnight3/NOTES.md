# Night 3: improvements (uncommitted)

Four improvements are applied to the working tree, each also saved here as a patch (2–5).
Every patch touches different files, so they apply independently to the committed code. Fire
(#1) and camera shake were built, then removed when you asked; nothing of either remains.

## 2. Fix for a rare deadlock with several worlds (robustness): `2_rapier_deadlock_fix.patch`
**Problem.** Tonight's test run hung. Rapier `rayon::spawn`s a deferred broad-phase task into
the pool the step runs in, then waits for it with a blocking `rx.recv()`, which parks a pool
worker. Rubble's physics pool is shared by every `World`, so when several worlds step at once,
workers can end up parked or waiting on parked ones.

**Change.** `rapier3d` 0.36.0 is now vendored the same way `parry3d` is (`vendor/rapier3d`; the
patch says to copy upstream 0.36.0 there first, then apply). One function changes: the wait
becomes `try_recv` plus `rayon::yield_now()`, so the waiting worker runs pending pool work,
including the task itself. Details are in `vendor/rapier3d/RUBBLE_PATCH.md`.
- I first tried a lock allowing one world in the pool at a time. It made 16 concurrent worlds 43%
  slower, so I dropped it.

**Evidence.**
- With the original Rapier, the `save_damage` tests hung in **7 of 25** runs. Patched: **25 of
  25** passed.
- Results are bit-identical: the four benchmark scenarios match night 2.
- 16 concurrent worlds take about the same time as before (14–16 s, against 16.4 s).
- Regression test: `tests/concurrent_worlds.rs`.

## 3. Visible damage (visuals): `3_damage_wear.patch`
- Chunks darken by up to 45% as they lose health.
- Damage is shown in 5 steps, and a chunk's render group is rebuilt only when it changes step, so
  steady damage costs a handful of rebuilds per chunk.

## 4. Sky (visuals): `4_sky.patch`
- Lighting mode (7) now draws Bevy's physically based atmosphere, lit by the same low sun: a warm
  horizon glow fading to grey-blue overhead, in place of the flat clear colour. It also provides
  the distance haze, so the old distance fog is off in this mode.
- The volumetric fog box is now wide enough that its sides no longer show as a seam in the sky.
- Plain mode is unchanged.

## 5. Blast waves blow out windows (physics + visuals): `5_blast_wave_windows.patch`
- Glass breaks at far lower pressure than walls, so an explosion now shatters panes out to
  `glass_blast_range` (3×) its radius when nothing solid (non-glass) is in the line between them.
  The line is aimed at the pane's centre; aiming at its nearest point grazed the window reveal.
  At most 512 panes are tested per blast.
- Result: a 6 m blast in front of office_1 blows out 27 panes, against 6 before.
- Tests: `tests/glass_blast.rs`. An exposed pane at 2 radii breaks, the same pane behind a
  concrete wall survives, nothing breaks beyond 3 radii, and the feature off is the control.

## Checks
- All tests in `rubble-core`, `rubble-stress`, `rubble-format` and `rubble-viewer` pass, except
  the old `stress_intact::real_buildings_stand_with_bending` (`office_xcoarse_3: 0.525`, same as
  before tonight).
- The fuzz test passes.
- Results are identical with 1 physics thread and with the default.
- No new clippy warnings.
- `rubble-sim`, `rubble-py` and `rubble-viewer` build.

## Behaviour changes to be aware of
- Explosions now break more windows. Set `glass_blast_range: 0` for the old behaviour.
- `Cargo.lock` now points `rapier3d` at the vendored copy.
