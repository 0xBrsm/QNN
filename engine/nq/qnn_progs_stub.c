/*
 * qnn_progs_stub.c — NQ-side stubs for QC VM evaluators.
 *
 * The QC VM evaluator (qw/qnn_progs.c) is built into the QW demo worker
 * only — it parses qwprogs.dat and tracks self.attack_finished across
 * cmd ticks to mirror the server's cooldown gate.  NQ-side builds (live
 * client, NQ demo worker, PPO worker) don't run this VM because they
 * either:
 *   - already see authoritative server state (live client), or
 *   - operate on .dem files where impulse / attack are observed signals
 *     rather than predicate inputs (NQ demo worker).
 *
 * The common ``qnn_self_common.c`` reads ``QNN_ProgsGetAttackCdRemainingSec``
 * to populate the new ``attack_finished`` self-token scalar.  On NQ
 * builds we return 0 (= "ready"), matching the historical regime where
 * cooldown state wasn't surfaced.
 *
 * If/when NQ grows a proper QC predicate evaluator, replace this stub
 * with the real implementation.
 */

float QNN_ProgsGetAttackCdRemainingSec(float now_seconds)
{
	(void)now_seconds;
	return 0.0f;
}

float QNN_ProgsGetAttackFinished(void)
{
	return 0.0f;
}

void QNN_ProgsSetAttackFinished(float value)
{
	(void)value;
}

/*
 * QNN_ProgsEvalAttack — NQ stub.
 *
 * The real QC-VM attack predicate lives in qw/qnn_progs.c and is built only
 * into the QW demo worker (it parses qwprogs.dat to mirror the server cooldown
 * gate). qnn_collect_helpers.c references it for the QWD operative-attack
 * feasibility column, but NQ builds (PPO worker / NQ demo worker) don't run
 * that VM. Return 0 (= "did not attack / reject"), matching the historical NQ
 * regime where the QC attack predicate wasn't evaluated. The PPO/eval runtime
 * does not exercise this collect-only path.
 */
int QNN_ProgsEvalAttack(
	float now_seconds,
	int health, int items_owned,
	int ammo_shells, int ammo_nails, int ammo_rockets, int ammo_cells,
	int weapon_id, int button0_pressed)
{
	(void)now_seconds; (void)health; (void)items_owned;
	(void)ammo_shells; (void)ammo_nails; (void)ammo_rockets; (void)ammo_cells;
	(void)weapon_id; (void)button0_pressed;
	return 0;
}
