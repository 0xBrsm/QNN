/*
 * qnn_weapon.c — per-weapon data tables, keyed by the RAW weapon id space
 * (1..8). See qnn_weapon.h for the two-id-space contract. All tables are
 * generated from QNN_WEAPON_LIST so a new weapon row stays consistent across
 * them, and they are reached only through the accessors (no exposed array to
 * mis-index with a subject-space id).
 */

#include "qnn_weapon.h"

/* Tables are positional, in QNN_WEAPON_LIST order, with index 0 = NONE.
 * Because qnn_weapon_id_t is generated from the same list in the same order,
 * table[QNN_WEAPON_X] == the X row by construction — no designated
 * initializers (C99) needed under -std=gnu89. */

static const float k_cooldown_sec[QNN_WEAPON_COUNT] = {
	0.0f,   /* QNN_WEAPON_NONE */
#define QNN_WEAPON_CD_ROW(e, label, cd) cd,
	QNN_WEAPON_LIST(QNN_WEAPON_CD_ROW)
#undef QNN_WEAPON_CD_ROW
};

static const char *const k_name[QNN_WEAPON_COUNT] = {
	"none",
#define QNN_WEAPON_NAME_ROW(e, label, cd) label,
	QNN_WEAPON_LIST(QNN_WEAPON_NAME_ROW)
#undef QNN_WEAPON_NAME_ROW
};

/* Compile-time guard: a missed/extra QNN_WEAPON_LIST row vs the enum count
 * makes one of these arrays the wrong size and the negative dimension fails
 * the build (C89-compatible static assert). */
typedef char qnn_weapon_cd_size_check[
	(sizeof(k_cooldown_sec) / sizeof(k_cooldown_sec[0]) == QNN_WEAPON_COUNT)
	? 1 : -1];
typedef char qnn_weapon_name_size_check[
	(sizeof(k_name) / sizeof(k_name[0]) == QNN_WEAPON_COUNT)
	? 1 : -1];

/* QNN_WeaponIsValid is a static inline in qnn_weapon.h. */

float QNN_WeaponCooldownSec(int weapon_id)
{
	return QNN_WeaponIsValid(weapon_id) ? k_cooldown_sec[weapon_id] : 0.0f;
}

const char *QNN_WeaponName(int weapon_id)
{
	if (weapon_id == QNN_WEAPON_NONE)
		return "none";
	return QNN_WeaponIsValid(weapon_id) ? k_name[weapon_id] : "?";
}
