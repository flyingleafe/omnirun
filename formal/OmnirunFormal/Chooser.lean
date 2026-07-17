/-
The pure slot chooser (DESIGN-V2 §2.1, policy SCHED-7): free-first, then the
cheapest affordable paid slot. "Cheapest" is positional — the input list is
the offer ranking already sorted by cost, so the first affordable element is
the cheapest affordable one.

Proved here:
* I8 (deadline-defense / free-first): if any free slot is offered, the
  chooser takes a free slot — a paid slot is never chosen while a fitting
  free one exists.
* Affordability soundness (COST-5 / per-job ceiling): a chosen paid slot is
  a member of the offer list and within the remaining budget.
* Never-refused-for-cost (SCHED-7): with a free slot present the chooser
  succeeds regardless of budget.
-/

namespace Omnirun

/-- Choose from cost-ranked `slots` given remaining budget `b`:
the first free slot, else the first (= cheapest) affordable one. -/
def chooseSlot (b : Nat) (slots : List Nat) : Option Nat :=
  match slots.find? (· == 0) with
  | some s => some s
  | none => slots.find? (· ≤ b)

/-- I8, free-first: a free slot in the offering means a free slot is chosen. -/
theorem chooseSlot_free_first (b : Nat) (slots : List Nat) (h : 0 ∈ slots) :
    chooseSlot b slots = some 0 := by
  unfold chooseSlot
  have hfind : slots.find? (· == 0) = some 0 := by
    induction slots with
    | nil => cases h
    | cons x xs ih =>
      by_cases hx0 : x = 0
      · subst hx0
        exact List.find?_cons_of_pos (by simp)
      · rw [List.find?_cons_of_neg (by simpa using hx0)]
        rcases List.mem_cons.mp h with h' | h'
        · exact absurd h'.symm hx0
        · exact ih h'
  rw [hfind]

/-- Soundness: whatever is chosen was offered, and a paid choice is
affordable (the per-job ceiling holds at choice time). -/
theorem chooseSlot_sound (b : Nat) (slots : List Nat) (c : Nat)
    (h : chooseSlot b slots = some c) : c ∈ slots ∧ (c = 0 ∨ c ≤ b) := by
  unfold chooseSlot at h
  cases hf : slots.find? (· == 0) with
  | some s =>
    rw [hf] at h
    cases h
    have hp := List.find?_some hf
    have hm := List.mem_of_find?_eq_some hf
    simp at hp
    exact ⟨hm, Or.inl hp⟩
  | none =>
    rw [hf] at h
    have hp := List.find?_some h
    have hm := List.mem_of_find?_eq_some h
    simp at hp
    exact ⟨hm, Or.inr hp⟩

/-- A job is never refused for cost: if a free slot exists the chooser
succeeds with it, whatever the budget (SCHED-7 run-late liveness). -/
theorem chooseSlot_never_refused (b : Nat) (slots : List Nat) (h : 0 ∈ slots) :
    ∃ c, chooseSlot b slots = some c ∧ c = 0 :=
  ⟨0, chooseSlot_free_first b slots h, rfl⟩

end Omnirun
