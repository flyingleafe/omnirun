/-
Executable semantics for the omnirun v2 kernel — Layer 2 of the trace
conformance scheme (docs/redesign/DESIGN-V2.md; skill: lean-trace-conformance).

`Action` is the model's alphabet: the implementation's `job_events` rows are
serialized 1:1 into these constructors. `apply` is the deterministic
executable transition function. The two theorems at the bottom transfer the
kernel's trust to the compiled trace checker:

  * `apply_sound`    : a step the checker accepts is a real `Step`
  * `apply_complete` : every real `Step` (from a well-formed state) is
                       accepted by the checker under its canonical action

Together with `preservation`/`reachable_inv` (Proofs.lean), a trace the
checker accepts end-to-end certifies that the implementation's run is a path
of the verified transition system.
-/
import OmnirunFormal.Model
import OmnirunFormal.Proofs

namespace Omnirun

/-- The trace alphabet. `submit` carries only the free fields of a fresh job
(id, committed cost) — the model forces the rest (queued, uncaptured,
unreaped, empty log). -/
inductive Action where
  | submit (id cost : Nat)
  | reserve (id : Nat)
  | provision (id : Nat)
  | activate (id : Nat)
  | rollback (id : Nat)
  | logAppend (id : Nat)
  | finish (id : Nat) (ok : Bool)
  | cancel (id : Nat)
  | fail (id : Nat)
  | capture (id : Nat)
  | reap (id : Nat)
  | releaseLost (id : Nat)
  | requeue (id : Nat)
  | unreachablePoll
deriving Repr, DecidableEq

/-- Locate a job by id, returning the surgical decomposition
`jobs = pre ++ j :: post`. -/
def findSplit (id : Nat) : List Job → Option (List Job × Job × List Job)
  | [] => none
  | j :: rest =>
    if j.id = id then some ([], j, rest)
    else
      match findSplit id rest with
      | some (pre, k, post) => some (j :: pre, k, post)
      | none => none

theorem findSplit_sound {id : Nat} :
    ∀ {l pre post : List Job} {j : Job},
      findSplit id l = some (pre, j, post) → l = pre ++ j :: post ∧ j.id = id := by
  intro l
  induction l with
  | nil => intro pre post j h; simp [findSplit] at h
  | cons a rest ih =>
    intro pre post j h
    unfold findSplit at h
    by_cases ha : a.id = id
    · rw [if_pos ha] at h
      cases h
      exact ⟨rfl, ha⟩
    · rw [if_neg ha] at h
      cases hf : findSplit id rest with
      | none => rw [hf] at h; cases h
      | some tri =>
        obtain ⟨pre', j', post'⟩ := tri
        rw [hf] at h
        cases h
        obtain ⟨hrest, hid⟩ := ih hf
        exact ⟨by rw [hrest]; rfl, hid⟩

/-- Under id-uniqueness the canonical split of `pre ++ j :: post` at `j.id`
is exactly `(pre, j, post)`. -/
theorem findSplit_complete :
    ∀ (pre : List Job) (j : Job) (post : List Job),
      ((pre ++ j :: post).map Job.id).Nodup →
      findSplit j.id (pre ++ j :: post) = some (pre, j, post) := by
  intro pre
  induction pre with
  | nil =>
    intro j post _
    simp [findSplit]
  | cons a pre ih =>
    intro j post hnd
    have hne : a.id ≠ j.id := by
      have := (ids_split (a :: pre) j post hnd).1 a (List.mem_cons.mpr (.inl rfl))
      exact this
    have hnd' : ((pre ++ j :: post).map Job.id).Nodup := by
      simp only [List.cons_append, List.map_cons] at hnd
      cases hnd with
      | cons _ h₂ => exact h₂
    show findSplit j.id (a :: (pre ++ j :: post)) = some (a :: pre, j, post)
    unfold findSplit
    rw [if_neg hne, ih j post hnd']

/-- Deterministic executable transition function. Guard structure mirrors the
`Step` constructors exactly. -/
def apply (m : M) : Action → Option M
  | .submit id cost =>
    if _h : id ∉ m.jobs.map Job.id then
      some { m with jobs := m.jobs ++ [⟨id, .queued, cost, false, false, 0⟩],
                    events := m.events + 1 }
    else none
  | .reserve id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .queued ∧ j.id ∉ m.intents ∧ m.active < m.cap ∧
              m.spent + j.cost ≤ m.budget then
        some { m with jobs := pre ++ { j with st := .placing } :: post,
                      intents := j.id :: m.intents,
                      spent := m.spent + j.cost,
                      events := m.events + 1 }
      else none
  | .provision id =>
    match findSplit id m.jobs with
    | none => none
    | some (_, j, _) =>
      if _h : j.st = .placing ∧ j.id ∈ m.intents ∧ j.id ∉ m.ext then
        some { m with ext := j.id :: m.ext, events := m.events + 1 }
      else none
  | .activate id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .placing ∧ j.id ∈ m.intents ∧ j.id ∈ m.ext then
        some { m with jobs := pre ++ { j with st := .placed } :: post,
                      intents := m.intents.erase j.id,
                      events := m.events + 1 }
      else none
  | .rollback id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .placing ∧ j.id ∈ m.intents ∧ j.id ∉ m.ext then
        some { m with jobs := pre ++ { j with st := .queued } :: post,
                      intents := m.intents.erase j.id,
                      spent := m.spent - j.cost,
                      events := m.events + 1 }
      else none
  | .logAppend id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .placed then
        some { m with jobs := pre ++ { j with logLen := j.logLen + 1 } :: post,
                      events := m.events + 1 }
      else none
  | .finish id ok =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .placed then
        some { m with jobs := pre ++ { j with st := if ok then .succeeded else .failed } :: post,
                      events := m.events + 1 }
      else none
  | .cancel id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st.terminal = false ∧ j.st ≠ .placing then
        some { m with jobs := pre ++ { j with st := .cancelled } :: post,
                      events := m.events + 1 }
      else none
  | .fail id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .queued then
        some { m with jobs := pre ++ { j with st := .failed } :: post,
                      events := m.events + 1 }
      else none
  | .capture id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .placed ∨ j.st.terminal = true then
        some { m with jobs := pre ++ { j with captured := true } :: post,
                      events := m.events + 1 }
      else none
  | .reap id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st.terminal = true ∧ j.captured = true then
        some { m with jobs := pre ++ { j with reaped := true } :: post,
                      ext := m.ext.erase j.id,
                      events := m.events + 1 }
      else none
  | .releaseLost id =>
    match findSplit id m.jobs with
    | none => none
    | some (_, j, _) =>
      if _h : j.st = .placed ∧ j.captured = true ∧ j.id ∈ m.ext then
        some { m with ext := m.ext.erase j.id, events := m.events + 1 }
      else none
  | .requeue id =>
    match findSplit id m.jobs with
    | none => none
    | some (pre, j, post) =>
      if _h : j.st = .placed ∧ j.id ∉ m.ext then
        some { m with jobs := pre ++ { j with st := .queued, captured := false, reaped := false } :: post,
                      spent := m.spent - j.cost,
                      events := m.events + 1 }
      else none
  | .unreachablePoll =>
    some { m with events := m.events + 1 }

/-! ### Soundness: an accepted step is a real `Step` -/

theorem apply_sound {m m' : M} {a : Action}
    (h : apply m a = some m') : Step m m' := by
  cases a with
  | submit id cost =>
    simp only [apply] at h
    split at h
    · cases h
      exact .submit m ⟨id, .queued, cost, false, false, 0⟩ (by assumption) rfl rfl rfl rfl
    · cases h
  | reserve id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .reserve m pre post j hj hg.1 hg.2.1 hg.2.2.1 hg.2.2.2
      · cases h
  | provision id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .provision m j (by rw [hj]; exact self_mem pre post)
          hg.1 hg.2.1 hg.2.2
      · cases h
  | activate id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .activate m pre post j hj hg.1 hg.2.1 hg.2.2
      · cases h
  | rollback id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .rollback m pre post j hj hg.1 hg.2.1 hg.2.2
      · cases h
  | logAppend id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .logAppend m pre post j hj hg
      · cases h
  | finish id ok =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .finish m pre post j ok hj hg
      · cases h
  | cancel id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .cancel m pre post j hj hg.1 hg.2
      · cases h
  | fail id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .failQueued m pre post j hj hg
      · cases h
  | capture id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .capture m pre post j hj hg
      · cases h
  | reap id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .reap m pre post j hj hg.1 hg.2
      · cases h
  | releaseLost id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .releaseLost m j (by rw [hj]; exact self_mem pre post)
          hg.1 hg.2.1 hg.2.2
      · cases h
  | requeue id =>
    simp only [apply] at h
    split at h
    · cases h
    · rename_i pre j post hf
      split at h
      · rename_i hg
        cases h
        obtain ⟨hj, _⟩ := findSplit_sound hf
        exact .requeue m pre post j hj hg.1 hg.2
      · cases h
  | unreachablePoll =>
    simp only [apply] at h
    cases h
    exact .unreachablePoll m

/-! ### Completeness: every real `Step` is accepted under its canonical
action -/

theorem apply_complete {m m' : M} (inv : Inv m) (h : Step m m') :
    ∃ a, apply m a = some m' := by
  cases h with
  | submit j hfresh hst hcf hrp hlog =>
    refine ⟨.submit j.id j.cost, ?_⟩
    simp only [apply]
    rw [dif_pos hfresh]
    have hje : (⟨j.id, .queued, j.cost, false, false, 0⟩ : Job) = j := by
      cases j
      simp_all
    rw [hje]
  | reserve pre post j hj hq hni hcap hbud =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.reserve j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨hq, hni, hcap, hbud⟩]
  | provision j hjm hpl hi hnx =>
    obtain ⟨pre, post, hj⟩ := List.append_of_mem hjm
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.provision j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨hpl, hi, hnx⟩]
  | activate pre post j hj hpl hi hx =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.activate j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨hpl, hi, hx⟩]
  | rollback pre post j hj hpl hi hnx =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.rollback j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨hpl, hi, hnx⟩]
  | logAppend pre post j hj hpl =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.logAppend j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos hpl]
  | finish pre post j ok hj hpl =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.finish j.id ok, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos hpl]
  | cancel pre post j hj hnt hnp =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.cancel j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨hnt, hnp⟩]
  | failQueued pre post j hj hq =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.fail j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos hq]
  | capture pre post j hj hp =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.capture j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos hp]
  | reap pre post j hj ht hc =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.reap j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨ht, hc⟩]
  | releaseLost j hjm hpl hc hx =>
    obtain ⟨pre, post, hj⟩ := List.append_of_mem hjm
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.releaseLost j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨hpl, hc, hx⟩]
  | requeue pre post j hj hpl hnx =>
    have hfs : findSplit j.id m.jobs = some (pre, j, post) := by
      rw [hj]
      exact findSplit_complete pre j post (hj ▸ inv.wf_ids)
    refine ⟨.requeue j.id, ?_⟩
    simp only [apply]
    split
    · rename_i hnone
      rw [hfs] at hnone
      cases hnone
    · rename_i pre' j' post' heq
      rw [hfs] at heq
      simp only [Option.some.injEq, Prod.mk.injEq] at heq
      obtain ⟨rfl, rfl, rfl⟩ := heq
      rw [dif_pos ⟨hpl, hnx⟩]
  | unreachablePoll =>
    exact ⟨.unreachablePoll, rfl⟩

/-- Trust transfer: a checker-accepted step preserves the invariant. -/
theorem apply_inv {m m' : M} {a : Action}
    (h : apply m a = some m') (inv : Inv m) : Inv m' :=
  preservation (apply_sound h) inv

end Omnirun
