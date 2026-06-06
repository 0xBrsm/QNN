/*
 * qnn_progs_stub.c — NQ-side stubs for QC VM evaluators.
 *
 * The QC VM evaluator (qw/qnn_progs.c) is built into the QW demo worker
 * only — it parses qwprogs.dat and tracks self.attack_finished across
 * cmd ticks to mirror the server's cooldown gate.  NQ-side builds (live
 * client, NQ demo worker, PPO worker) don't run this VM because they
 * either:
 *   - already see authoritative server state (live client), or
 *   - operate on .dem files where impulse / fire are observed signals
 *     rather than predicate inputs (NQ demo worker).
 *
 * The common ``qnn_self_common.c`` reads ``QNN_ProgsGetAttackCdRemaining``
 * to populate the new ``attack_finished`` self-token scalar.  On NQ
 * builds we return 0 (= "ready"), matching the historical regime where
 * cooldown state wasn't surfaced.
 *
 * If/when NQ grows a proper QC predicate evaluator, replace this stub
 * with the real implementation.
 */

int QNN_ProgsGetAttackCdRemaining(int tick, int tick_hz)
{
	(void)tick;
	(void)tick_hz;
	return 0;
}
