# Overnight engine improvements (uncommitted)

Three changes to `rubble-core`, already applied to the working tree. Each is also saved here as
its own patch, stacked in order: applied to the code as it was before tonight, 1 → 2 → 3 gives
exactly the current tree. Undo one with, for example, `git apply -R overnight/3_crush_front.patch`
(undo them in reverse order). Each is a `WorldConfig` setting, so it can also be switched off
without touching code.

## 1. Fast debris no longer passes through slabs: `1_adaptive_ccd.patch`
- **Problem:** a known limit. Pieces faster than about 12 m/s passed through 0.2 m slabs. Also,
  `ccd: true` did nothing, because CCD was never enabled on any body.
- **Change:** small pieces (`ccd_max_chunks` = 16 or fewer chunks) moving faster than
  `ccd_speed` (9 m/s) get Rapier's soft (predictive) CCD, with a look-ahead of one step's travel,
  re-checked every tick. `ccd: true` now really sweeps every body.
- **Why not full CCD:** switching on Rapier's full CCD made collapse ticks 2–4× slower, and
  switching it on globally costs that much even when no body uses it.
- **Cost:** office 2.46 → 2.23 ms per tick, tower 1.47 → 1.53, apartment 4.00 → 3.99.
- **Tests:** `tests/tunneling.rs`. A block dropped from 60 m (33 m/s) and 150 m (54 m/s) is now
  caught by a 0.2 m slab. A control test shows it still passes through with the setting off.

## 2. Explosions scatter settled rubble: `2_blast_rubble.patch`
- **Problem:** frozen rubble ignored blast impulses. A grenade in a rubble pile damaged chunks but
  never moved any.
- **Change:** rubble pushed harder than `blast_thaw_dv` (0.5 m/s) thaws into moving pieces that
  start with the velocity the blast gives them, and refreezes later. New bodies have no mass until
  the next physics step, so the push becomes their starting velocity; promotion already did this,
  and that code is now a shared helper with identical results.
- **Result:** in a blast inside a settled office rubble pile, 1,834 chunks move, against 110 before.
- **Tests:** `tests/blast_rubble.rs`. A weak, distant blast still leaves the rubble frozen.

## 3. Tall sections crush themselves down floor by floor: `3_crush_front.patch`
- **Problem:** a known limit. building4's upper 47 floors dropped one storey and stood there.
- **Change:** a "crush front". When a section taller than `crush_min_extent` (8 m) lands with more
  energy than the storey-deep band at its base can absorb, that band is crushed. 90% of it is
  pulverised (shattered); `crush_debris_every` = 10 keeps the rest as debris. The section keeps
  falling with the energy left over:
  ½mv_out² = ½mv_in² + m·g·δ − E_crush,
  where E_crush = (capacity of the band's vertical bonds) × `crush_distance` (0.5 m).
- **Impact speed:** taken from the piece's velocity recorded before the landing started
  (`Cluster::v_quiet`, sampled before each physics step). It is not current velocity minus the
  contact impulses: once the section sinks into its own debris, the solver's push-out impulses
  look like extra impact and fed energy in (in the first version, the section ran away to
  60 m/s).
- **Result (first version, measured on building4 before you deleted it):** with storey 5 cut,
  18 storeys crushed in a row. After 10 s the highest piece was at 17.9 m, against 180.9 m
  before. Average tick time went from 4.8 to 2.8 ms.
- **Result (current version, 30-storey synthetic tube):** the crush runs down 25 storeys,
  speeding up smoothly from 7.7 to about 29 m/s, and never faster than free fall.
- **Unchanged:** the office and apartment collapses never pass the energy gate, so their results
  are identical to before.
- **Tests:** `tests/crush_front.rs`, on a stiff synthetic tube tower, since no remaining asset is
  tall enough. They check that the crush runs all the way down, that the roof never falls
  faster than free fall (this fails with the old speed estimate), that an impossible crush
  energy gives exactly the old result, and that a short section gives exactly the old result.
- **Demo:** `building4_crush_front.gif`, recorded with the first version before building4 was
  deleted.

## Checks
- All `rubble-core`, `rubble-stress`, `rubble-format` and `rubble-viewer` tests pass, except
  `stress_intact::real_buildings_stand_with_bending`. That test fails identically on the code from
  before tonight, so it is unrelated. Now that building4 is gone it only reports
  `office_xcoarse_3`.
- The fuzz test (`fuzz_nothing_hangs`, ignored by default) reports 0 violations on all 7 buildings.
- Results are deterministic: identical outcomes with 1 physics thread and with the default.
- No new clippy warnings.
- The Python tests were not run: they need the extension rebuilt into the virtual environment.

## Caveats
- On building4, pressing X left the tower standing. X's wedge cut drops a tower only about 0.8 m,
  which is below the crush threshold, as it should be. A cut of about one storey starts the crush.
- Most of a crushed tower is pulverised. That is cheap and looks right with particles on, but it
  means the rubble pile is smaller than the building.
- The "Known limits" section of `docs/explainer.html` (fast debris tunnelling, tall sections
  staying rigid) is now out of date. I have not edited it.
