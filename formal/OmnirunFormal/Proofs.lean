/-
Invariants of the omnirun v2 kernel and their proofs.

Mapping to DESIGN-V2 §10 (and REQUIREMENTS.md):
  I1  budget-safety            → Inv.budget_ok, preserved by `preservation`
  I2  concurrency-safety       → Inv.cap_ok
  I3  no-silent-loss           → `no_stuck` (+ `reserve_enabled`)
  I4  cancellation-completeness→ `terminal_absorbing` (terminal never mutates)
  I5  no-untracked-money       → Inv.ext_tracked (+ Inv.ext_nodup: ≤1 per key)
  I6  capture-before-release   → Inv.reap_captured (reap/releaseLost guards)
  I7  effectively-once         → Inv.ext_nodup + deterministic keys (create
                                 guarded by absence; adopt otherwise)
  I8  deadline-defense         → Chooser.chooseSlot_free_first
  I9  convergence              → `reserve_convergent`
  I10 unreachable-freeze       → `unreachable_changes_nothing`
  I11 event-fold consistency   → `events_strict_mono` (every transition
                                 appends exactly one event)
  I12 log-monotonicity         → `log_monotone`
-/
import OmnirunFormal.Model

namespace Omnirun

/-! ### List helper lemmas -/

theorem active_surgery (pre post : List Job) (j : Job) :
    (pre ++ j :: post).countP (fun k => k.st.activeB)
      = pre.countP (fun k => k.st.activeB)
        + post.countP (fun k => k.st.activeB)
        + (if j.st.activeB then 1 else 0) := by
  simp [List.countP_append, List.countP_cons]
  omega

theorem active_snoc (l : List Job) (j : Job) :
    (l ++ [j]).countP (fun k => k.st.activeB)
      = l.countP (fun k => k.st.activeB) + (if j.st.activeB then 1 else 0) := by
  simp [List.countP_append, List.countP_cons]

theorem mem_surgery {j k : Job} {pre post : List Job}
    (hk : k ∈ pre ++ j :: post) : k ∈ pre ∨ k = j ∨ k ∈ post := by
  rcases List.mem_append.mp hk with h | h
  · exact .inl h
  · rcases List.mem_cons.mp h with h | h
    · exact .inr (.inl h)
    · exact .inr (.inr h)

theorem side_mem {j' k : Job} {pre post : List Job}
    (h : k ∈ pre ∨ k ∈ post) : k ∈ pre ++ j' :: post := by
  rcases h with h | h
  · exact List.mem_append.mpr (.inl h)
  · exact List.mem_append.mpr (.inr (List.mem_cons.mpr (.inr h)))

theorem self_mem {j' : Job} (pre post : List Job) :
    j' ∈ pre ++ j' :: post :=
  List.mem_append.mpr (.inr (List.mem_cons.mpr (.inl rfl)))

/-- Under id-uniqueness, elements beside the surgical position carry a
different id. -/
theorem ids_split :
    ∀ (pre : List Job) (j : Job) (post : List Job),
      ((pre ++ j :: post).map Job.id).Nodup →
      (∀ k ∈ pre, k.id ≠ j.id) ∧ (∀ k ∈ post, k.id ≠ j.id) := by
  intro pre
  induction pre with
  | nil =>
    intro j post h
    simp only [List.nil_append, List.map_cons] at h
    cases h with
    | cons h₁ _ =>
      refine ⟨fun k hk => absurd hk (List.not_mem_nil), ?_⟩
      intro k hk he
      exact h₁ k.id (List.mem_map.mpr ⟨k, hk, rfl⟩) he.symm
  | cons a pre ih =>
    intro j post h
    simp only [List.cons_append, List.map_cons] at h
    cases h with
    | cons h₁ h₂ =>
      obtain ⟨hpre, hpost⟩ := ih j post h₂
      refine ⟨?_, hpost⟩
      intro k hk
      rcases List.mem_cons.mp hk with rfl | hk'
      · intro he
        exact h₁ j.id
          (List.mem_map.mpr ⟨j, self_mem pre post, rfl⟩) he
      · exact hpre k hk'

theorem wf_ids_surgery {pre post : List Job} {j j' : Job}
    (h : ((pre ++ j :: post).map Job.id).Nodup) (hid : j'.id = j.id) :
    ((pre ++ j' :: post).map Job.id).Nodup := by
  simpa [List.map_append, List.map_cons, hid] using h

theorem nodup_snoc {l : List Nat} {a : Nat}
    (h : l.Nodup) (ha : a ∉ l) : (l ++ [a]).Nodup := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    cases h with
    | cons h₁ h₂ =>
      have hax : a ≠ x := fun he => ha (he ▸ List.mem_cons.mpr (.inl rfl))
      have ha' : a ∉ xs := fun hm => ha (List.mem_cons.mpr (.inr hm))
      refine List.Pairwise.cons ?_ (ih h₂ ha')
      intro b hb
      rcases List.mem_append.mp hb with hb | hb
      · exact h₁ b hb
      · rcases List.mem_cons.mp hb with rfl | hb
        · exact fun he => hax he.symm
        · exact absurd hb (List.not_mem_nil)

/-- Same-shape helper for lists of jobs keyed by id. -/
theorem nodup_snoc_jobs {l : List Job} {j : Job}
    (h : (l.map Job.id).Nodup) (hj : j.id ∉ l.map Job.id) :
    ((l ++ [j]).map Job.id).Nodup := by
  rw [List.map_append]
  exact nodup_snoc h hj

/-- Pointwise property transfer across a surgical update. -/
theorem pointwise_surgery {P : Job → Prop} {m : M} {pre post : List Job}
    {j j' : Job} (hj : m.jobs = pre ++ j :: post)
    (hall : ∀ k ∈ m.jobs, P k) (hj' : P j') :
    ∀ k ∈ pre ++ j' :: post, P k := by
  intro k hk
  rcases mem_surgery hk with hs | rfl | hs
  · exact hall k (by rw [hj]; exact side_mem (.inl hs))
  · exact hj'
  · exact hall k (by rw [hj]; exact side_mem (.inr hs))

/-- Every provider-side resource is store-tracked: an open intent or a job
record past QUEUED under the same key (I5). -/
def TrackedIn (jobs : List Job) (intents : List Nat) (rid : Nat) : Prop :=
  rid ∈ intents ∨ ∃ j, j ∈ jobs ∧ j.id = rid ∧ j.st ≠ .queued

/-- Tracking survives a surgical job update whose new state is not QUEUED
(or is unchanged), under any intent-set enlargement. -/
theorem tracked_mono {pre post : List Job} {j j' : Job}
    {ints ints' : List Nat} {rid : Nat}
    (h : TrackedIn (pre ++ j :: post) ints rid)
    (hid : j'.id = j.id)
    (hcase : j'.st ≠ .queued ∨ j'.st = j.st)
    (hIm : ∀ r ∈ ints, r ∈ ints') :
    TrackedIn (pre ++ j' :: post) ints' rid := by
  rcases h with hI | ⟨k, hk, hkid, hkst⟩
  · exact .inl (hIm _ hI)
  · rcases mem_surgery hk with hs | rfl | hs
    · exact .inr ⟨k, side_mem (.inl hs), hkid, hkst⟩
    · rcases hcase with hne | hsame
      · exact .inr ⟨j', self_mem pre post, hid.trans hkid, hne⟩
      · exact .inr ⟨j', self_mem pre post, hid.trans hkid, by rw [hsame]; exact hkst⟩
    · exact .inr ⟨k, side_mem (.inr hs), hkid, hkst⟩

/-! ### The inductive invariant -/

structure Inv (m : M) : Prop where
  wf_ids : (m.jobs.map Job.id).Nodup
  /-- I1 — spend never exceeds the budget window cap. -/
  budget_ok : m.spent ≤ m.budget
  /-- I2 — capacity-occupying placements never exceed the provider cap. -/
  cap_ok : m.active ≤ m.cap
  /-- I7 — at most one provider resource per deterministic job key. -/
  ext_nodup : m.ext.Nodup
  /-- I5 — no untracked billable resource, ever. -/
  ext_tracked : ∀ rid ∈ m.ext, TrackedIn m.jobs m.intents rid
  /-- I6 — a confirmed release implies artifacts were captured first. -/
  reap_captured : ∀ j ∈ m.jobs, j.reaped = true → j.captured = true
  /-- A placing job always has its write-ahead intent open (recovery can
  always resolve it — part of I3/I7). -/
  placing_intent : ∀ j ∈ m.jobs, j.st = .placing → j.id ∈ m.intents

theorem inv_init (budget cap : Nat) : Inv (M.init budget cap) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [M.init, M.active_def]

/-! ### Preservation: every kernel transition maintains the invariant -/

theorem preservation {m m' : M} (h : Step m m') (inv : Inv m) : Inv m' := by
  cases h with
  | submit j hfresh hst hcf hrp hlog =>
    refine ⟨nodup_snoc_jobs inv.wf_ids hfresh, inv.budget_ok, ?_, inv.ext_nodup,
            ?_, ?_, ?_⟩
    · show (m.jobs ++ [j]).countP (fun k => k.st.activeB) ≤ m.cap
      rw [active_snoc]
      have hc := inv.cap_ok
      rw [M.active_def] at hc
      simp [hst]
      omega
    · intro rid hr
      rcases inv.ext_tracked rid hr with hI | ⟨k, hk, hkid, hkst⟩
      · exact .inl hI
      · exact .inr ⟨k, List.mem_append.mpr (.inl hk), hkid, hkst⟩
    · intro k hk
      rcases List.mem_append.mp hk with hk | hk
      · exact inv.reap_captured k hk
      · rcases List.mem_cons.mp hk with rfl | h'
        · intro hr; rw [hrp] at hr; cases hr
        · exact absurd h' (List.not_mem_nil)
    · intro k hk hkst
      rcases List.mem_append.mp hk with hk | hk
      · exact inv.placing_intent k hk hkst
      · rcases List.mem_cons.mp hk with rfl | h'
        · rw [hst] at hkst; cases hkst
        · exact absurd h' (List.not_mem_nil)
  | reserve pre post j hj hq hni hcap hbud =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    have hcaplt := hcap
    rw [M.active_def, hj, active_surgery] at hcaplt
    refine ⟨wf_ids_surgery hids rfl, hbud, ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with st := .placing } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      simp [hq] at hcaplt
      simp
      omega
    · intro rid hr
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      exact tracked_mono hold rfl (.inl (by simp)) (fun r hrI => List.mem_cons.mpr (.inr hrI))
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      exact inv.reap_captured j (by rw [hj]; exact self_mem pre post)
    · intro k hk hkp
      rcases mem_surgery hk with hs | rfl | hs
      · exact List.mem_cons.mpr (.inr (inv.placing_intent k
          (by rw [hj]; exact side_mem (.inl hs)) hkp))
      · exact List.mem_cons.mpr (.inl rfl)
      · exact List.mem_cons.mpr (.inr (inv.placing_intent k
          (by rw [hj]; exact side_mem (.inr hs)) hkp))
  | provision j hjm hpl hi hnx =>
    refine ⟨inv.wf_ids, inv.budget_ok, inv.cap_ok, ?_, ?_, inv.reap_captured,
            inv.placing_intent⟩
    · exact List.Pairwise.cons (fun b hb he => hnx (he ▸ hb)) inv.ext_nodup
    · intro rid hr
      rcases List.mem_cons.mp hr with rfl | hr'
      · exact .inl hi
      · exact inv.ext_tracked rid hr'
  | activate pre post j hj hpl hi hx =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hsplit := ids_split pre j post hids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, inv.budget_ok, ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with st := .placed } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      simp [hpl] at hcap'
      simp
      omega
    · intro rid hr
      by_cases hrid : rid = j.id
      · subst hrid
        exact .inr ⟨{ j with st := .placed }, self_mem pre post, rfl, by simp⟩
      · have hold := inv.ext_tracked rid hr
        rw [hj] at hold
        rcases hold with hI | ⟨k, hk, hkid, hkst⟩
        · exact .inl ((List.mem_erase_of_ne hrid).mpr hI)
        · rcases mem_surgery hk with hs | rfl | hs
          · exact .inr ⟨k, side_mem (.inl hs), hkid, hkst⟩
          · exact absurd hkid.symm hrid
          · exact .inr ⟨k, side_mem (.inr hs), hkid, hkst⟩
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      exact inv.reap_captured j (by rw [hj]; exact self_mem pre post)
    · intro k hk hkp
      rcases mem_surgery hk with hs | rfl | hs
      · exact (List.mem_erase_of_ne (hsplit.1 k hs)).mpr
          (inv.placing_intent k (by rw [hj]; exact side_mem (.inl hs)) hkp)
      · simp at hkp
      · exact (List.mem_erase_of_ne (hsplit.2 k hs)).mpr
          (inv.placing_intent k (by rw [hj]; exact side_mem (.inr hs)) hkp)
  | rollback pre post j hj hpl hi hnx =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hsplit := ids_split pre j post hids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, Nat.le_trans (Nat.sub_le _ _) inv.budget_ok,
            ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with st := .queued } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      simp [hpl] at hcap'
      simp
      omega
    · intro rid hr
      have hrid : rid ≠ j.id := fun he => hnx (he ▸ hr)
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      rcases hold with hI | ⟨k, hk, hkid, hkst⟩
      · exact .inl ((List.mem_erase_of_ne hrid).mpr hI)
      · rcases mem_surgery hk with hs | rfl | hs
        · exact .inr ⟨k, side_mem (.inl hs), hkid, hkst⟩
        · exact absurd hkid.symm hrid
        · exact .inr ⟨k, side_mem (.inr hs), hkid, hkst⟩
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      exact inv.reap_captured j (by rw [hj]; exact self_mem pre post)
    · intro k hk hkp
      rcases mem_surgery hk with hs | rfl | hs
      · exact (List.mem_erase_of_ne (hsplit.1 k hs)).mpr
          (inv.placing_intent k (by rw [hj]; exact side_mem (.inl hs)) hkp)
      · simp at hkp
      · exact (List.mem_erase_of_ne (hsplit.2 k hs)).mpr
          (inv.placing_intent k (by rw [hj]; exact side_mem (.inr hs)) hkp)
  | logAppend pre post j hj hpl =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, inv.budget_ok, ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with logLen := j.logLen + 1 } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      have hco : (if ({ j with logLen := j.logLen + 1 } : Job).st.activeB then 1 else 0)
          = (if j.st.activeB then 1 else 0) := rfl
      omega
    · intro rid hr
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      exact tracked_mono hold rfl (.inr rfl) (fun r hrI => hrI)
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      exact inv.reap_captured j (by rw [hj]; exact self_mem pre post)
    · intro k hk hkp
      refine pointwise_surgery (P := fun k => k.st = .placing → k.id ∈ m.intents)
        hj inv.placing_intent ?_ k hk hkp
      intro hp
      simp [hpl] at hp
  | finish pre post j ok hj hpl =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, inv.budget_ok, ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with st := if ok then .succeeded else .failed } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      have : (if ({ j with st := if ok then .succeeded else .failed } : Job).st.activeB
          then 1 else 0) = 0 := by cases ok <;> simp
      omega
    · intro rid hr
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      refine tracked_mono hold rfl (.inl ?_) (fun r hrI => hrI)
      cases ok <;> simp
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      exact inv.reap_captured j (by rw [hj]; exact self_mem pre post)
    · intro k hk hkp
      refine pointwise_surgery (P := fun k => k.st = .placing → k.id ∈ m.intents)
        hj inv.placing_intent ?_ k hk hkp
      intro hp
      cases ok <;> simp at hp
  | cancel pre post j hj hnt hnp =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, inv.budget_ok, ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with st := .cancelled } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      simp
      omega
    · intro rid hr
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      exact tracked_mono hold rfl (.inl (by simp)) (fun r hrI => hrI)
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      exact inv.reap_captured j (by rw [hj]; exact self_mem pre post)
    · intro k hk hkp
      refine pointwise_surgery (P := fun k => k.st = .placing → k.id ∈ m.intents)
        hj inv.placing_intent ?_ k hk hkp
      intro hp
      simp at hp
  | failQueued pre post j hj hq =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, inv.budget_ok, ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with st := .failed } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      simp
      omega
    · intro rid hr
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      exact tracked_mono hold rfl (.inl (by simp)) (fun r hrI => hrI)
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      exact inv.reap_captured j (by rw [hj]; exact self_mem pre post)
    · intro k hk hkp
      refine pointwise_surgery (P := fun k => k.st = .placing → k.id ∈ m.intents)
        hj inv.placing_intent ?_ k hk hkp
      intro hp
      simp at hp
  | capture pre post j hj hp =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, inv.budget_ok, ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with captured := true } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      have hco : (if ({ j with captured := true } : Job).st.activeB then 1 else 0)
          = (if j.st.activeB then 1 else 0) := rfl
      omega
    · intro rid hr
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      exact tracked_mono hold rfl (.inr rfl) (fun r hrI => hrI)
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      intro _
      rfl
    · intro k hk hkp
      refine pointwise_surgery (P := fun k => k.st = .placing → k.id ∈ m.intents)
        hj inv.placing_intent ?_ k hk hkp
      intro hpl
      have : j.st = .placing := hpl
      rcases hp with hp | hp
      · rw [this] at hp; cases hp
      · rw [this] at hp; simp at hp
  | reap pre post j hj ht hc =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, inv.budget_ok, ?_, ?_, ?_, ?_, ?_⟩
    · show (pre ++ { j with reaped := true } :: post).countP
          (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      have hco : (if ({ j with reaped := true } : Job).st.activeB then 1 else 0)
          = (if j.st.activeB then 1 else 0) := rfl
      omega
    · exact List.Sublist.nodup List.erase_sublist inv.ext_nodup
    · intro rid hr
      have hr' := List.mem_of_mem_erase hr
      have hold := inv.ext_tracked rid hr'
      rw [hj] at hold
      exact tracked_mono hold rfl (.inr rfl) (fun r hrI => hrI)
    · intro k hk
      refine pointwise_surgery
        (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      intro _
      exact hc
    · intro k hk hkp
      refine pointwise_surgery (P := fun k => k.st = .placing → k.id ∈ m.intents)
        hj inv.placing_intent ?_ k hk hkp
      intro hpl
      have hplj : j.st = .placing := hpl
      rw [hplj] at ht
      simp at ht
  | releaseLost j hjm hpl hc hx =>
    refine ⟨inv.wf_ids, inv.budget_ok, inv.cap_ok, ?_, ?_, inv.reap_captured,
            inv.placing_intent⟩
    · exact List.Sublist.nodup List.erase_sublist inv.ext_nodup
    · intro rid hr
      exact inv.ext_tracked rid (List.mem_of_mem_erase hr)
  | requeue pre post j hj hpl hnx =>
    have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
    have hcap' := inv.cap_ok
    rw [M.active_def, hj, active_surgery] at hcap'
    refine ⟨wf_ids_surgery hids rfl, Nat.le_trans (Nat.sub_le _ _) inv.budget_ok,
            ?_, inv.ext_nodup, ?_, ?_, ?_⟩
    · show (pre ++ { j with st := .queued, captured := false, reaped := false }
            :: post).countP (fun k => k.st.activeB) ≤ m.cap
      rw [active_surgery]
      simp [hpl] at hcap'
      simp
      omega
    · intro rid hr
      have hrid : rid ≠ j.id := fun he => hnx (he ▸ hr)
      have hold := inv.ext_tracked rid hr
      rw [hj] at hold
      rcases hold with hI | ⟨k, hk, hkid, hkst⟩
      · exact .inl hI
      · rcases mem_surgery hk with hs | rfl | hs
        · exact .inr ⟨k, side_mem (.inl hs), hkid, hkst⟩
        · exact absurd hkid.symm hrid
        · exact .inr ⟨k, side_mem (.inr hs), hkid, hkst⟩
    · intro k hk
      refine pointwise_surgery (P := fun k => k.reaped = true → k.captured = true)
        hj inv.reap_captured ?_ k hk
      intro hr
      cases hr
    · intro k hk hkp
      refine pointwise_surgery (P := fun k => k.st = .placing → k.id ∈ m.intents)
        hj inv.placing_intent ?_ k hk hkp
      intro hp
      simp at hp
  | unreachablePoll =>
    exact ⟨inv.wf_ids, inv.budget_ok, inv.cap_ok, inv.ext_nodup,
           inv.ext_tracked, inv.reap_captured, inv.placing_intent⟩

/-! ### Reachability: the invariant holds in every reachable state -/

inductive Reachable : M → Prop where
  | init (budget cap : Nat) : Reachable (M.init budget cap)
  | step {m m' : M} : Reachable m → Step m m' → Reachable m'

theorem reachable_inv {m : M} (h : Reachable m) : Inv m := by
  induction h with
  | init b c => exact inv_init b c
  | step _ hs ih => exact preservation hs ih

/-! ### I4 — terminal states are absorbing (cancellation completeness) -/

theorem terminal_absorbing {m m' : M} (h : Step m m') :
    ∀ j ∈ m.jobs, j.st.terminal = true →
      ∃ j' ∈ m'.jobs, j'.id = j.id ∧ j'.st = j.st := by
  cases h with
  | submit j0 hfresh hst hcf hrp hlog =>
    intro k hk _
    exact ⟨k, List.mem_append.mpr (.inl hk), rfl, rfl⟩
  | reserve pre post j0 hj hq hni hcap hbud =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · rw [hq] at ht; simp at ht
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | provision j0 hjm hpl hi hnx =>
    intro k hk _; exact ⟨k, hk, rfl, rfl⟩
  | activate pre post j0 hj hpl hi hx =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · rw [hpl] at ht; simp at ht
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | rollback pre post j0 hj hpl hi hnx =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · rw [hpl] at ht; simp at ht
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | logAppend pre post j0 hj hpl =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · exact ⟨{ k with logLen := k.logLen + 1 }, self_mem pre post, rfl, rfl⟩
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | finish pre post j0 ok hj hpl =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · rw [hpl] at ht; simp at ht
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | cancel pre post j0 hj hnt hnp =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · rw [ht] at hnt; cases hnt
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | failQueued pre post j0 hj hq =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · rw [hq] at ht; simp at ht
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | capture pre post j0 hj hp =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · exact ⟨{ k with captured := true }, self_mem pre post, rfl, rfl⟩
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | reap pre post j0 hj ht0 hc =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · exact ⟨{ k with reaped := true }, self_mem pre post, rfl, rfl⟩
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | releaseLost j0 hjm hpl hc hx =>
    intro k hk _; exact ⟨k, hk, rfl, rfl⟩
  | requeue pre post j0 hj hpl hnx =>
    intro k hk ht
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, rfl⟩
    · rw [hpl] at ht; simp at ht
    · exact ⟨k, side_mem (.inr hs), rfl, rfl⟩
  | unreachablePoll =>
    intro k hk _; exact ⟨k, hk, rfl, rfl⟩

/-! ### I12 — the durable log only ever grows, across every transition
(including requeue: attempts accumulate) -/

theorem log_monotone {m m' : M} (h : Step m m') :
    ∀ j ∈ m.jobs, ∃ j' ∈ m'.jobs, j'.id = j.id ∧ j.logLen ≤ j'.logLen := by
  cases h with
  | submit j0 hfresh hst hcf hrp hlog =>
    intro k hk
    exact ⟨k, List.mem_append.mpr (.inl hk), rfl, Nat.le_refl _⟩
  | reserve pre post j0 hj hq hni hcap hbud =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with st := .placing }, self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | provision j0 hjm hpl hi hnx =>
    intro k hk; exact ⟨k, hk, rfl, Nat.le_refl _⟩
  | activate pre post j0 hj hpl hi hx =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with st := .placed }, self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | rollback pre post j0 hj hpl hi hnx =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with st := .queued }, self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | logAppend pre post j0 hj hpl =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with logLen := k.logLen + 1 }, self_mem pre post, rfl,
        Nat.le_succ _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | finish pre post j0 ok hj hpl =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with st := if ok then .succeeded else .failed },
        self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | cancel pre post j0 hj hnt hnp =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with st := .cancelled }, self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | failQueued pre post j0 hj hq =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with st := .failed }, self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | capture pre post j0 hj hp =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with captured := true }, self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | reap pre post j0 hj ht hc =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with reaped := true }, self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | releaseLost j0 hjm hpl hc hx =>
    intro k hk; exact ⟨k, hk, rfl, Nat.le_refl _⟩
  | requeue pre post j0 hj hpl hnx =>
    intro k hk
    rcases mem_surgery (hj ▸ hk) with hs | rfl | hs
    · exact ⟨k, side_mem (.inl hs), rfl, Nat.le_refl _⟩
    · exact ⟨{ k with st := .queued, captured := false, reaped := false },
        self_mem pre post, rfl, Nat.le_refl _⟩
    · exact ⟨k, side_mem (.inr hs), rfl, Nat.le_refl _⟩
  | unreachablePoll =>
    intro k hk; exact ⟨k, hk, rfl, Nat.le_refl _⟩

/-! ### I11 — every transition appends exactly one event -/

theorem events_strict_mono {m m' : M} (h : Step m m') :
    m.events < m'.events := by
  cases h <;> exact Nat.lt_succ_self _

/-! ### I10 — an unreachable backend changes nothing -/

theorem unreachable_changes_nothing (m : M) :
    ∃ m', Step m m' ∧ m'.jobs = m.jobs ∧ m'.ext = m.ext ∧
          m'.intents = m.intents ∧ m'.spent = m.spent :=
  ⟨_, .unreachablePoll m, rfl, rfl, rfl, rfl⟩

/-! ### I3 — no stuck states: every non-terminal job has an enabled step -/

theorem no_stuck {m : M} (inv : Inv m) {j : Job} (hjm : j ∈ m.jobs)
    (hnt : j.st.terminal = false) : ∃ m', Step m m' := by
  obtain ⟨pre, post, hsplit⟩ := List.append_of_mem hjm
  cases hst : j.st with
  | queued =>
    exact ⟨_, .cancel m pre post j hsplit hnt (by rw [hst]; decide)⟩
  | placing =>
    have hi := inv.placing_intent j hjm hst
    by_cases hx : j.id ∈ m.ext
    · exact ⟨_, .activate m pre post j hsplit hst hi hx⟩
    · exact ⟨_, .rollback m pre post j hsplit hst hi hx⟩
  | placed =>
    exact ⟨_, .cancel m pre post j hsplit hnt (by rw [hst]; decide)⟩
  | succeeded => rw [hst] at hnt; cases hnt
  | failed => rw [hst] at hnt; cases hnt
  | cancelled => rw [hst] at hnt; cases hnt

/-- A queued job with capacity and budget can always be reserved (the
scheduling half of I3). -/
theorem reserve_enabled {m : M} {j : Job} (hjm : j ∈ m.jobs)
    (hq : j.st = .queued) (hni : j.id ∉ m.intents)
    (hcap : m.active < m.cap) (hbud : m.spent + j.cost ≤ m.budget) :
    ∃ m', Step m m' := by
  obtain ⟨pre, post, hsplit⟩ := List.append_of_mem hjm
  exact ⟨_, .reserve m pre post j hsplit hq hni hcap hbud⟩

/-! ### I9 — convergence: enacting a reserve disables re-reserving the same
job, so an identical second scheduling pass places nothing new -/

theorem reserve_convergent {m : M} {pre post : List Job} {j : Job}
    (inv : Inv m) (hj : m.jobs = pre ++ j :: post) :
    ∀ k ∈ pre ++ { j with st := .placing } :: post,
      k.id = j.id → k.st ≠ .queued := by
  have hids : ((pre ++ j :: post).map Job.id).Nodup := hj ▸ inv.wf_ids
  have hids' := wf_ids_surgery (j' := { j with st := .placing }) hids rfl
  have hsplit := ids_split pre { j with st := .placing } post hids'
  intro k hk hkid
  rcases mem_surgery hk with hs | rfl | hs
  · exact absurd hkid (hsplit.1 k hs)
  · intro hq; simp at hq
  · exact absurd hkid (hsplit.2 k hs)

end Omnirun
