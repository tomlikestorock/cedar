/*
 * Copyright Cedar Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! This file defines the algorithm for emitting well-formedness assumptions about
//! Cedar hierarchies. A valid Cedar hierarchy graph is the irreflexive transitive
//! closure of a DAG.  (Note that the `in` operator is reflexive, but this is
//! enforced separately by the semantics, and it's not needed in the hierarchy.)
//!
//! The generated assumptions are expressed as Terms that enforce the acyclicty and
//! transitivity constraints on the symbolic store, for a given Cedar expression.
//!
//! For more technical details, see comments in SymCC/Enforcer.lean.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use cedar_policy_core::ast::{Expr, ExprKind};

use super::{
    compiler::compile,
    env::{SymEntities, SymEnv},
    factory::{and, app, implies, is_some, not, option_get, set_is_empty, set_member, set_subset},
    function::UnaryFunction,
    term::Term,
    term_type::TermType,
    type_abbrevs::EntityType,
};

/// Helper function used by `footprint()`.
/// In Lean this is defined inline inside `footprint()`, but in Rust, you can't
/// define closures that take generic arguments (like `impl IntoIterator`), so
/// we define it as a standalone function.
fn of_branch<'a>(
    x1: &Expr,
    ft1: impl IntoIterator<Item = Term> + 'a,
    ft2: impl IntoIterator<Item = Term> + 'a,
    ft3: impl IntoIterator<Item = Term> + 'a,
    env: &SymEnv,
) -> Box<dyn Iterator<Item = Term> + 'a> {
    match compile(x1, env) {
        Ok(Term::Some(t)) if *t == true.into() => Box::new(ft2.into_iter()),
        Ok(Term::Some(t)) if *t == false.into() => Box::new(ft3.into_iter()),
        Ok(_) => Box::new(ft1.into_iter().chain(ft2).chain(ft3)),
        Err(_) => Box::new(std::iter::empty()),
    }
}

/// Returns the terms corresponding to subexpressions of `x` of the following form:
///
///   * A variable term with an entity type
///   * An entity reference literal
///   * An attribute access expression with an entity type
///   * A binary (`getTag`) expression with an entity type
///
/// These are the only basic expressions in Cedar that may evaluate to an entity.
/// All other expressions that evaluate to an entity are build up from the above
/// basic expressions.
///
/// All returned terms are of type `TermType.option .entity`.
//
// NOTE: this essentially calls `compile()` on every subexpression, which is
// redundant by itself, and also potentially redundant with other `compile()`
// calls in verifier.rs. For a much more optimized version, use functionality
// in the `symccopt` module.
#[expect(
    unreachable_patterns,
    reason = "pattern may or may not be unreachable depending on whether `tolerant-ast` feature of cedar-policy-core is enabled"
)]
pub(crate) fn footprint<'a>(x: &'a Expr, env: &'a SymEnv) -> Box<dyn Iterator<Item = Term> + 'a> {
    let of_entity = |x: &Expr| -> Box<dyn Iterator<Item = Term>> {
        match compile(x, env) {
            Ok(t) => {
                if t.type_of().is_option_entity_type() {
                    Box::new(std::iter::once(t))
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Err(_) => Box::new(std::iter::empty()),
        }
    };
    #[expect(clippy::unimplemented, reason = "Should fail at an earlier stage")]
    match x.expr_kind() {
        ExprKind::Lit(_) | ExprKind::Var(_) => of_entity(x),
        ExprKind::If {
            test_expr,
            then_expr,
            else_expr,
        } => of_branch(
            test_expr,
            footprint(test_expr, env),
            footprint(then_expr, env),
            footprint(else_expr, env),
            env,
        ),
        ExprKind::And { left, right } => of_branch(
            left,
            footprint(left, env),
            footprint(right, env),
            std::iter::empty(),
            env,
        ),
        ExprKind::Or { left, right } => of_branch(
            left,
            footprint(left, env),
            std::iter::empty(),
            footprint(right, env),
            env,
        ),
        ExprKind::BinaryApp { arg1, arg2, .. } => Box::new(
            of_entity(x)
                .chain(footprint(arg1, env))
                .chain(footprint(arg2, env)),
        ),
        ExprKind::GetAttr { expr, .. } => Box::new(of_entity(x).chain(footprint(expr, env))),
        ExprKind::HasAttr { expr, .. }
        | ExprKind::UnaryApp { arg: expr, .. }
        | ExprKind::Like { expr, .. }
        | ExprKind::Is { expr, .. } => footprint(expr, env),
        ExprKind::ExtensionFunctionApp { args: exprs, .. } | ExprKind::Set(exprs) => {
            Box::new(exprs.iter().flat_map(|x| footprint(x, env)))
        }
        ExprKind::Record(axs) => Box::new(axs.values().flat_map(|x| footprint(x, env))),
        ExprKind::Slot(_) => unimplemented!("analyzing templates is not currently supported"),
        ExprKind::Unknown(_) => {
            unimplemented!("analyzing partial expressions is not currently supported")
        }
        _ => unimplemented!("analyzing `{}` is not currently supported", x),
    }
}

/// Returns the set of Terms corresponding to the footprints of `exprs`.
/// Returning a `BTreeSet` ensures there are no duplicates.
pub(crate) fn footprints<'a>(
    exprs: impl IntoIterator<Item = &'a Expr>,
    env: &SymEnv,
) -> BTreeSet<Term> {
    exprs.into_iter().flat_map(|x| footprint(x, env)).collect()
}

/// Returns the acyclicity constraint for the given term
pub(crate) fn acyclicity(t: &Term, es: &SymEntities) -> Term {
    match t.type_of() {
        TermType::Option { ty } if matches!(*ty, TermType::Entity { .. }) => {
            match Arc::unwrap_or_clone(ty) {
                TermType::Entity { ety } => match es.ancestors_of_type(&ety, &ety) {
                    Some(f) => {
                        let t_unwrapped = option_get(t.clone());
                        implies(
                            is_some(t.clone()),
                            not(set_member(t_unwrapped.clone(), app(f.clone(), t_unwrapped))),
                        )
                    }
                    None => true.into(),
                },
                #[expect(
                    clippy::unreachable,
                    reason = "Code already checks that matches entity_type"
                )]
                _ => unreachable!("already checked it matches TermType::Entity above"),
            }
        }
        _ => true.into(),
    }
}

/// Returns the transitivity constraint for the given term
pub(crate) fn transitivity(t1: &Term, t2: &Term, es: &SymEntities) -> Term {
    let is_ancestor = |t2_unwrapped: Term, t1_unwrapped: Term, f12: UnaryFunction| -> Term {
        and(
            and(is_some(t1.clone()), is_some(t2.clone())),
            set_member(t2_unwrapped, app(f12, t1_unwrapped)),
        )
    };
    let are_ancestors = |t2_unwrapped: &Term,
                         anc2: &BTreeMap<EntityType, UnaryFunction>,
                         t1_unwrapped: &Term,
                         ety1|
     -> Term {
        anc2.iter().fold(true.into(), |acc, (ety3, f23)| {
            // inlining Lean's `areAncestorsOfType` here
            let are_ancestors_of_type = match es.ancestors_of_type(ety1, ety3) {
                Some(f13) => set_subset(
                    app(f23.clone(), t2_unwrapped.clone()),
                    app(f13.clone(), t1_unwrapped.clone()),
                ),
                None => set_is_empty(app(f23.clone(), t2_unwrapped.clone())),
            };
            and(acc, are_ancestors_of_type)
        })
    };

    if t1 == t2 {
        true.into()
    } else {
        match (t1.type_of(), t2.type_of()) {
            (TermType::Option { ty: ty1 }, TermType::Option { ty: ty2 }) => {
                match (Arc::unwrap_or_clone(ty1), Arc::unwrap_or_clone(ty2)) {
                    (TermType::Entity { ety: ety1 }, TermType::Entity { ety: ety2 }) => {
                        match (es.ancestors_of_type(&ety1, &ety2), es.ancestors(&ety2)) {
                            (Some(f12), Some(anc2)) => {
                                let t1_unwrapped = option_get(t1.clone());
                                let t2_unwrapped = option_get(t2.clone());
                                implies(
                                    is_ancestor(
                                        t2_unwrapped.clone(),
                                        t1_unwrapped.clone(),
                                        f12.clone(),
                                    ),
                                    are_ancestors(&t2_unwrapped, anc2, &t1_unwrapped, &ety1),
                                )
                            }
                            (_, _) => true.into(),
                        }
                    }
                    (_, _) => true.into(),
                }
            }
            (_, _) => true.into(),
        }
    }
}

/// Whether every ancestor function of `t`'s entity type is a concrete `.udf`
/// table rather than an uninterpreted `.uuf` symbol. Mirrors cedar-lean `allUdf`.
///
/// Written as explicit early-returns rather than a nested `match` so each branch
/// states what holds, without reasoning about the negation.
fn all_udf(es: &SymEntities, t: &Term) -> bool {
    // Non-optional or non-entity terms carry no ancestor table, so they assert
    // nothing about ancestors and are never inert.
    let TermType::Option { ty } = t.type_of() else {
        return false;
    };
    let TermType::Entity { ety } = ty.as_ref() else {
        return false;
    };
    // A type with no ancestor table has nothing uninterpreted to assert.
    let Some(anc) = es.ancestors(ety) else {
        return true;
    };
    // All-udf iff no ancestor function is an uninterpreted `.uuf`.
    anc.values().all(|f| matches!(f, UnaryFunction::Udf(_)))
}

/// A footprint term is *inert* when it is a literal with only concrete (`.udf`)
/// ancestor tables, so its transitivity smart-constructs to `true` and `enforce`
/// skips it. Mirrors cedar-lean `inert`.
fn inert(es: &SymEntities, t: &Term) -> bool {
    // INVARIANT: the `is_literal` guard is load-bearing — a non-literal all-udf
    // term still yields a symbolic `ite` from `app`, so it must not be pruned.
    t.is_literal() && all_udf(es, t)
}

/// Returns the ground acyclicity and transitivity assumptions for `xs` and `env`.
///
/// Skips transitivity over *inert* footprint pairs (see `inert`) to avoid the
/// O(N²) growth in emitted transitivity terms. Sound and complete per cedar-lean
/// `transitive_implies_Transitive_udf_udf` and `mem_transitivePruned_mem_transitive`.
pub fn enforce<'a>(xs: impl IntoIterator<Item = &'a Expr>, env: &SymEnv) -> BTreeSet<Term> {
    let ts = footprints(xs, env);
    let es = &env.entities;

    // Tag each footprint term with whether it is inert (see `inert`), hoisting
    // the per-term half of the test out of the N² loop below.
    let tagged: Vec<(&Term, bool)> = ts.iter().map(|t| (t, inert(es, t))).collect();

    let mut out: BTreeSet<Term> = ts.iter().map(|t| acyclicity(t, es)).collect();
    // This nested loop is N² in the footprint size. If that ever becomes a
    // bottleneck, replace it with a linear pass that only pairs each kept
    // (non-inert) term with the others, instead of walking every pair.
    for &(t1, inert1) in &tagged {
        for &(t2, inert2) in &tagged {
            // Both inert: `app` on their concrete tables makes transitivity
            // smart-construct to `true`, so skip the pair (the emitted Set is
            // unchanged modulo that `true` — see this fn's doc).
            if inert1 && inert2 {
                continue;
            }
            // Otherwise the pair may carry a real constraint, so emit it.
            out.insert(transitivity(t1, t2, es));
        }
    }
    out
}
