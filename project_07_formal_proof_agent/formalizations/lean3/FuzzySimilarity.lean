/- Project 07 / Phase 07: portable finite-grid fuzzy Jaccard formalization. -/

namespace project07

/-
fuzzy_member encodes the exact membership grid {0, 1/4, 1/2, 3/4, 1}.
similarity_value encodes every exact Jaccard value obtained on that grid.
This Lean 3 source deliberately uses ASCII syntax and no imports so that the
pinned Windows Lean 3.42.1 toolchain does not depend on a source-file code page.
-/
inductive fuzzy_member
| zero : fuzzy_member
| quarter : fuzzy_member
| half : fuzzy_member
| three_quarters : fuzzy_member
| one : fuzzy_member

inductive similarity_value
| zero : similarity_value
| quarter : similarity_value
| third : similarity_value
| half : similarity_value
| two_thirds : similarity_value
| three_quarters : similarity_value
| one : similarity_value

/- Exact membership value carried by each finite-grid constructor. -/
def membership_value : fuzzy_member -> similarity_value
| fuzzy_member.zero := similarity_value.zero
| fuzzy_member.quarter := similarity_value.quarter
| fuzzy_member.half := similarity_value.half
| fuzzy_member.three_quarters := similarity_value.three_quarters
| fuzzy_member.one := similarity_value.one

/-
Exact singleton fuzzy Jaccard table on the denominator-four membership grid.
The empty-union convention is the first case, J(0,0) = 1.
-/
def fuzzy_jaccard : fuzzy_member -> fuzzy_member -> similarity_value
| fuzzy_member.zero fuzzy_member.zero := similarity_value.one
| fuzzy_member.zero fuzzy_member.quarter := similarity_value.zero
| fuzzy_member.zero fuzzy_member.half := similarity_value.zero
| fuzzy_member.zero fuzzy_member.three_quarters := similarity_value.zero
| fuzzy_member.zero fuzzy_member.one := similarity_value.zero
| fuzzy_member.quarter fuzzy_member.zero := similarity_value.zero
| fuzzy_member.quarter fuzzy_member.quarter := similarity_value.one
| fuzzy_member.quarter fuzzy_member.half := similarity_value.half
| fuzzy_member.quarter fuzzy_member.three_quarters := similarity_value.third
| fuzzy_member.quarter fuzzy_member.one := similarity_value.quarter
| fuzzy_member.half fuzzy_member.zero := similarity_value.zero
| fuzzy_member.half fuzzy_member.quarter := similarity_value.half
| fuzzy_member.half fuzzy_member.half := similarity_value.one
| fuzzy_member.half fuzzy_member.three_quarters := similarity_value.two_thirds
| fuzzy_member.half fuzzy_member.one := similarity_value.half
| fuzzy_member.three_quarters fuzzy_member.zero := similarity_value.zero
| fuzzy_member.three_quarters fuzzy_member.quarter := similarity_value.third
| fuzzy_member.three_quarters fuzzy_member.half := similarity_value.two_thirds
| fuzzy_member.three_quarters fuzzy_member.three_quarters := similarity_value.one
| fuzzy_member.three_quarters fuzzy_member.one := similarity_value.three_quarters
| fuzzy_member.one fuzzy_member.zero := similarity_value.zero
| fuzzy_member.one fuzzy_member.quarter := similarity_value.quarter
| fuzzy_member.one fuzzy_member.half := similarity_value.half
| fuzzy_member.one fuzzy_member.three_quarters := similarity_value.three_quarters
| fuzzy_member.one fuzzy_member.one := similarity_value.one

/- The explicit empty-union convention: J(0,0) is exactly 1. -/
theorem fuzzy_jaccard_zero_zero :
  fuzzy_jaccard fuzzy_member.zero fuzzy_member.zero = similarity_value.one := rfl

/- Every registered grid membership is reflexive. -/
theorem fuzzy_jaccard_refl (a : fuzzy_member) :
  fuzzy_jaccard a a = similarity_value.one :=
by cases a; refl

/- The finite-grid singleton Jaccard relation is symmetric. -/
theorem fuzzy_jaccard_symm (a b : fuzzy_member) :
  fuzzy_jaccard a b = fuzzy_jaccard b a :=
by cases a; cases b; refl

/- The witness members are exactly 1/4 and 1/2. -/
theorem quarter_has_exact_value :
  membership_value fuzzy_member.quarter = similarity_value.quarter := rfl

theorem half_has_exact_value :
  membership_value fuzzy_member.half = similarity_value.half := rfl

/- The counterexample similarity is exactly 1/2. -/
theorem fuzzy_jaccard_counterexample_value :
  fuzzy_jaccard fuzzy_member.quarter fuzzy_member.half = similarity_value.half := rfl

/- The counterexample cannot be similarity 1. -/
theorem fuzzy_jaccard_counterexample_not_one :
  fuzzy_jaccard fuzzy_member.quarter fuzzy_member.half = similarity_value.one -> false :=
by intro h; cases h

end project07
