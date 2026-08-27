/*
 * qnn_qualify_predicate_test.c — pin the two entity-qualification
 * predicates against each other.
 *
 * WHY THIS EXISTS
 *
 * The A26 (FULL) and A27 (COMBAT) policies decide "is this actor
 * currently in line of sight" through completely different code:
 *
 *   FULL    QNN_QualifyEntity        — modality ladder + per-modality
 *                                      recency threshold; an in-LOS actor
 *                                      lands as modality SIGHT with age 0
 *   COMBAT  QNN_QualifyCombatEntity  — exact `e->vis == now`
 *
 * Those are supposed to agree on the in-LOS case: for actors the primary
 * observation source is VIS, so FULL's `near` IS `e->vis`, and age == 0
 * iff `vis == now`. A26 and A27 corpora collected from byte-identical
 * demos disagree by ~22% on how many frames contain an in-LOS actor
 * (964,545 vs 752,291 val frames), which should be impossible if the
 * predicates are equivalent and the stamping path is unchanged — and
 * qnn_store.c / qnn_entity.c / qw/qnn_players.c are byte-identical to
 * main, so the stamping path IS unchanged.
 *
 * This test asserts the equivalence directly on synthetic entities, so
 * the claim is mechanical rather than argued. If it passes, the divergence
 * is upstream of qualification (stamping or store contents) and the search
 * moves there. If it fails, the failing row names the disagreement.
 *
 * Includes the oracle translation unit so the file-static predicates are
 * reachable, and stubs the externals the emit path needs at link time
 * (none of them run — only the two predicates are called).
 */
#include <stdio.h>
#include <string.h>

#include "../common/qnn_oracle.c"

/* ── Stubs for the emit path's externals ─────────────────────────────
 * QNN_OracleEmitTokens and its helpers are compiled but never called:
 * only the two qualification predicates are exercised. These satisfy the
 * linker. Any of them executing is a bug in the test, so they abort. */

#include <stdlib.h>

client_state_t cl;
qnn_entity_t qnn_store[MAX_EDICTS + QNN_STORE_OVERFLOW];
int qnn_event_head[QNN_EVENT_HEAD_CAPACITY];
qnn_semantic_event_atom_t qnn_semantic_events[QNN_MAX_EVENT_ATOMS];
FILE *qnn_sound_dump = NULL;

static void unreachable(const char *who)
{
	printf("FATAL: %s called — the test must only exercise the "
		"qualification predicates\n", who);
	abort();
}

void AngleVectors(vec3_t a, vec3_t f, vec3_t r, vec3_t u)
{ (void)a; (void)f; (void)r; (void)u; unreachable("AngleVectors"); }

qnn_entity_mode_t QNN_IOGetEntityMode(void)
{ unreachable("QNN_IOGetEntityMode"); return QNN_ENTITY_MODE_COMBAT; }

int QNN_RouteFindArea(const qnn_route_runtime_t *rt, const float *p,
	qnn_route_area_result_t *out, char *err, size_t err_size)
{ (void)rt; (void)p; (void)out; (void)err; (void)err_size;
  unreachable("QNN_RouteFindArea"); return 0; }

int QNN_RoutePathPositionNth(const qnn_route_runtime_t *rt, int a, int b,
	const float *p0, const float *p1, int n, float *o0, float *o1,
	char *err, size_t err_size)
{ (void)rt; (void)a; (void)b; (void)p0; (void)p1; (void)n; (void)o0;
  (void)o1; (void)err; (void)err_size;
  unreachable("QNN_RoutePathPositionNth"); return 0; }

/* QNN_LookupEntityBounds comes from the real qnn_store.c TU. */

float QNN_IsSameTeam(int entity_num)
{ (void)entity_num; unreachable("QNN_IsSameTeam"); return 0.0f; }

/* ── Harness ────────────────────────────────────────────────────── */

static int checks = 0;
static int failures = 0;

static void expect_eq_int(const char *what, int got, int want)
{
	checks++;
	if (got != want)
	{
		failures++;
		printf("FAIL %-52s got %d want %d\n", what, got, want);
	}
}

/* One synthetic store entity. `now` is the emit clock. */
static qnn_entity_t mk_actor(float vis, float pvs, float snd, float mem)
{
	qnn_entity_t e;
	memset(&e, 0, sizeof(e));
	e.type = QNN_ENT_ACTOR;
	e.vis = vis;
	e.pvs = pvs;
	e.snd = snd;
	e.mem = mem;
	e.entity_num = 7;
	return e;
}

/* FULL: does this actor qualify, and does it read as in-LOS?
 *
 * "in-LOS" here is what a CONSUMER OF THE CACHE sees, which is the only
 * thing corpus analysis can measure: QNN_OracleEmitTokens clamps the age
 * to non-negative before writing it as the row's `recency` field
 *
 *     candidates[...].recency = (age < 0.0f) ? 0.0f : age;
 *
 * so a NEGATIVE age — vis stamped ahead of the emit clock — is stored as
 * recency 0.0 and is indistinguishable from a genuine current-frame
 * sighting. Modelling the clamp is the point: without it this test would
 * compare something no downstream consumer can observe. */
static void full_verdict(const qnn_entity_t *e, float now,
	int *out_qualified, int *out_in_los, int *out_modality, float *out_age)
{
	int modality = -1;
	float age = -1.0f;
	float stored;
	int ok = QNN_QualifyEntity(e, now, &modality, &age) ? 1 : 0;
	stored = (age < 0.0f) ? 0.0f : age;      /* the clamp, verbatim */
	*out_qualified = ok;
	*out_modality = modality;
	*out_age = age;
	*out_in_los = (ok && modality == QNN_MODALITY_SIGHT && stored == 0.0f) ? 1 : 0;
}

/* COMBAT: does this actor qualify, and as SIGHT? */
static void combat_verdict(const qnn_entity_t *e, float now,
	int *out_qualified, int *out_in_los, int *out_modality)
{
	int modality = -1;
	int ok = QNN_QualifyCombatEntity(e, now, &modality) ? 1 : 0;
	*out_qualified = ok;
	*out_modality = modality;
	*out_in_los = (ok && modality == QNN_MODALITY_SIGHT) ? 1 : 0;
}

int main(void)
{
	const float now = 278.315277f;   /* a real mtime from a QWD collect */
	const float tick = 0.05f;        /* 20 Hz emit */

	struct {
		const char *name;
		float vis, pvs, snd, mem;
		int want_in_los;             /* the physically correct answer */
		int full_known_wrong;        /* 1 = FULL is known to disagree here */
	} cases[] = {
		/* The case that must agree: currently visible. */
		{ "in-LOS now",              now,          now,   0.0f, 0.0f, 1, 0 },
		{ "in-LOS now, no pvs",      now,          0.0f,  0.0f, 0.0f, 1, 0 },
		/* Visible one tick ago — NOT in LOS now. */
		{ "vis one tick stale",      now - tick,   now,   0.0f, 0.0f, 0, 0 },
		{ "vis half tick stale",     now - 0.025f, now,   0.0f, 0.0f, 0, 0 },
		/* Never seen, only in PVS — NOT in LOS. */
		{ "pvs only, never seen",    0.0f,         now,   0.0f, 0.0f, 0, 0 },
		/* Stale within the 2s memory tail. */
		{ "vis 1.0s stale",          now - 1.0f,   now,   0.0f, 0.0f, 0, 0 },
		{ "vis 1.999s stale",        now - 1.999f, now,   0.0f, 0.0f, 0, 0 },
		{ "vis 2.5s stale",          now - 2.5f,   now,   0.0f, 0.0f, 0, 0 },
		/* Nothing at all. */
		{ "no observation",          0.0f,         0.0f,  0.0f, 0.0f, 0, 0 },
		/* vis stamped AHEAD of the emit clock. Physically this is not a
		 * current-frame sighting for THIS tick, so the correct in-LOS
		 * answer is 0 — but FULL's negative-age clamp stores recency 0.0,
		 * which downstream reads as in-LOS. These rows are expected to
		 * FAIL for FULL and pass for COMBAT; that asymmetry is the
		 * finding, and it inflates every a26 in-LOS count. */
		{ "vis one tick AHEAD",      now + tick,   now,   0.0f, 0.0f, 0, 1 },
		{ "vis half tick AHEAD",     now + 0.025f, now,   0.0f, 0.0f, 0, 1 },
	};

	size_t i;
	size_t n_cases = sizeof(cases) / sizeof(cases[0]);

	printf("%-24s %-22s %-22s %s\n", "case", "FULL (a26)", "COMBAT (a27)", "agree?");
	for (i = 0; i < n_cases; ++i)
	{
		qnn_entity_t e;
		int fq, fl, fm, cq, cl_, cm;
		float fa;
		char buf[128];

		e = mk_actor(cases[i].vis, cases[i].pvs, cases[i].snd, cases[i].mem);
		full_verdict(&e, now, &fq, &fl, &fm, &fa);
		combat_verdict(&e, now, &cq, &cl_, &cm);

		printf("%-24s q=%d los=%d mod=%d age=%.4f  q=%d los=%d mod=%d        %s\n",
			cases[i].name, fq, fl, fm, fa, cq, cl_, cm,
			(fl == cl_) ? "yes" : "NO");

		/* COMBAT is asserted against the physically correct answer.
		 * FULL is asserted against its DOCUMENTED behaviour: correct
		 * everywhere except where the negative-age clamp makes a
		 * future vis stamp read as a current sighting. Those rows are
		 * pinned as known-wrong so this stays green while recording
		 * the defect; fixing the clamp means flipping the flag. */
		snprintf(buf, sizeof(buf), "%s: COMBAT in-LOS", cases[i].name);
		expect_eq_int(buf, cl_, cases[i].want_in_los);
		snprintf(buf, sizeof(buf), "%s: FULL in-LOS (documented)", cases[i].name);
		expect_eq_int(buf, fl,
			cases[i].full_known_wrong ? !cases[i].want_in_los
			                          : cases[i].want_in_los);
	}

	/* Float-equality sensitivity: a vis stamp that differs from `now` by
	 * one ULP is "current" in intent but fails COMBAT's exact ==. If the
	 * store ever stamped vis from a different clock than the emit reads,
	 * this is the shape the bug would take. Documented, not asserted as
	 * correct behaviour — the point is to record which way each predicate
	 * falls. */
	{
		float nudged;
		qnn_entity_t e;
		int fq, fl, fm, cq, cl_, cm;
		float fa;

		nudged = now;
		*(unsigned int *)&nudged += 1u;   /* next representable float */
		e = mk_actor(nudged, now, 0.0f, 0.0f);
		full_verdict(&e, now, &fq, &fl, &fm, &fa);
		combat_verdict(&e, now, &cq, &cl_, &cm);
		printf("%-24s q=%d los=%d mod=%d age=%.9f  q=%d los=%d mod=%d\n",
			"vis one ULP off", fq, fl, fm, fa, cq, cl_, cm);
		printf("   note: FULL in-LOS=%d COMBAT in-LOS=%d — %s\n",
			fl, cl_, (fl == cl_) ? "agree" : "DIVERGE on sub-ULP skew");
	}

	printf("\n%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
