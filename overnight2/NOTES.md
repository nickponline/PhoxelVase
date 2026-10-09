# Night 2: five engine improvements (uncommitted)

All five are applied to the working tree. Each is also saved here as its own patch, stacked in
order: applied to the code as it was before tonight, 1 → 5 gives exactly the current tree. Undo
one with `git apply -R overnight2/<n>_*.patch`, undoing in reverse order.

## 1. Faster rubble support check (performance): `1_support_check_early_exit.patch`
- **Problem:** to decide whether frozen rubble is still supported, the engine probed every chunk
  of a group (2–3 collision queries each) and only then checked whether the supported area
  holds the centre of mass. Big fallen slabs cost thousands of queries per check. In the
  apartment collapse this was 1,550 ms of the 2,000 ms total.
- **Change:** compute the centre of mass first, then test the supported area as it grows (at 4,
  8, 16, … supported chunks) and stop as soon as it holds. The area only grows, so the answer
  is exactly the same.
- **Result:**
  - Apartment collapse support check: 1,550 → 76 ms.
  - Apartment collapse tick: 4.04 → 1.03 ms on average, slowest 1% 11.3 → 3.2 ms.
  - `rubble-sim bench` collapse: 20.4 → 3.5 ms per tick (worst 118 → 22 ms).
- **Checks:** every scenario ends bit-identical (same events, same final state).

## 2. Impact events (feature): `2_impact_events.patch`
- **Change:** new `Event::Impact { building, pos, normal, impulse, speed, material }`, fired
  once per new hard contact (bounces count as new contacts; resting stays silent). It is meant
  for sounds, camera shake and dust. `normal` points from the surface to the moving piece, and
  `speed` is impulse ÷ the lighter mass, so it doesn't depend on size. Pairs are sorted, so the
  order is deterministic.
- **Settings:** `impact_event_min` (1000 N·s; 0 = off) and `impact_events_max` (64 per tick,
  hardest first).
- **Viewer:** uses these events instead of its own second contact scan every tick.
- **Cost:** about 1–4% of collapse tick time.
- **Tests:** `tests/impact_events.rs`. A 4.8 m drop reports a landing at about 8.6 m/s with an
  upward normal and concrete material, then weaker bounces, then nothing once it rests.

## 3. Support sweep skips settled rubble (performance): `3_support_sweep_skip.patch`
- **Problem:** the safety-net sweep kept re-checking rubble that rests on static things. That
  support can only disappear when a collider is removed, and every removal already re-checks its
  surroundings.
- **Change:** such groups are marked (`Building::support_static`) and the sweep skips them. They
  still count against the sweep's budget, so all other groups are checked on exactly the same
  ticks as before.
- **Result:** `rubble-sim bench` collapse 3.54 → 1.28 ms per tick on average (slowest 1%
  6.0 → 3.7 ms). Together with #1 that is 20.4 → 1.28 ms (16×).
- **Checks:** all scenarios bit-identical (with impact events off, to compare against the
  baseline). The support tests and the fuzz test pass.

## 4. Focus-aware debris budgets (feature): `4_focus_debris_budget.patch`
- **API:** `World::set_focus(&[points])`, for example the player cameras.
- **Behaviour:**
  - When the piece budget is exceeded, small debris farthest from every focus point goes first.
    Big pieces are never dropped early for being far away.
  - Debris farther than `lod_far` (60 m) lives `lod_far_ttl` (¼) as long.
  - At the cap, debris in view of a focus point can still be created.
- **No focus set:** exactly as before.
- **Viewer:** sets the camera position as the focus every frame.
- **Tests:** `tests/focus.rs`. Two walls 120 m apart are blasted together with a 30-piece cap:
  focus on A gives 30 pieces at A and 0 at B; focus on B gives 0 and 30; no focus gives 29 and
  0 (the wall processed first takes every slot). Far debris is also gone by 4 s.

## 5. Save and restore building damage (feature): `5_save_restore_damage.patch`
- **API:** `World::save_damage(b) -> DamageState` (serde: destroyed chunks, broken joints,
  damaged hit points and joint health, rubble groups with poses relative to the building) and
  `World::restore_damage(b, &state)`.
- **Behaviour:**
  - Restoring applies the state to a freshly loaded copy of the building, without events or
    debris.
  - Pieces still moving at save time are saved as rubble where they are.
  - The state is checked against the building it is applied to.
- **Use:** save games, levels that stay damaged, streaming buildings in and out.
- **Tests:** `tests/save_damage.rs`, all on office_1.
  - A blasted ruin saves as 37 KB of JSON and restores exactly: every chunk in the same state
    and position (within 0.1 mm), and the same joints.
  - The restored ruin stays put.
  - It follows a building loaded elsewhere and rotated.
  - It refuses the wrong building, or a second restore.
  - A fully collapsed office (2,573 rubble chunks) restores with only 2 chunks shifting more
    than 5 cm, from pieces that were still settling when it was saved.

## Checks
- All tests in `rubble-core`, `rubble-stress`, `rubble-format` and `rubble-viewer` pass, except
  `stress_intact::real_buildings_stand_with_bending`. It fails the same way on tonight's
  starting code (`office_xcoarse_3: 0.525`), so it is unrelated.
- The fuzz test (`fuzz_nothing_hangs`, ignored by default) passes.
- Results are identical with 1 physics thread and with the default.
- No new clippy warnings. `rubble-sim`, `rubble-py` and `rubble-viewer` all build.

## Looked at, not changed
- **Toppling and hinges:** office_1 with half its ground floor destroyed stays up as a 12 m
  cantilever (1,811 chunks). A four-storey facade can act as a deep beam, so this is a judgement
  call about the structural model; I left it for you.
- **Stress catch-up after big damage:** a known limit in `docs/explainer.html`, but a catch-up
  solve already exists (`catchup_frac`), so the explainer is out of date there.
- **First-tick stress solve (40 ms for 91,000 chunks):** already runs in parallel across
  buildings.
