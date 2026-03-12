#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int step_count = 0;
static int episode_seed = -1;
static char map_id[64] = "UNKNOWN";
static int tick_hz = 20;
static int previous_look_yaw = 12;
static int previous_look_pitch = 12;
static int previous_fire = 0;
static int previous_jump = 0;
static int previous_weapon = 0;
static int reset_maxplayers = 1;
static int reset_deathmatch = 0;
static int reset_teamplay = 0;
static int current_frags = 0;
static int current_monster_kills = 0;
static int kill_event_step = -1;
static int total_damage_dealt = 0;
static int total_hit_count = 0;
static int total_shots_fired = 0;
static int weapon_damage_dealt[9] = {0};
static int weapon_hits_landed[9] = {0};
static int weapon_shots_fired[9] = {0};
static int binary_step_enabled = 0;

#define STEP_BINARY_MAGIC "QWLD"
#define STEP_BINARY_VERSION 1

#define STEP_FLAG_DONE 0x0001
#define STEP_FLAG_GOAL_REACHED 0x0002

#define EVENT_FLAG_HAS_DELTA 0x0001
#define EVENT_FLAG_HAS_WEAPON_ID 0x0002

#define DONE_REASON_NONE 0
#define DONE_REASON_GOAL_REACHED 1

#define EVENT_DAMAGE_TAKEN 1
#define EVENT_PICKUP_HEALTH 2
#define EVENT_PICKUP_ARMOR 3
#define EVENT_PICKUP_AMMO 4
#define EVENT_PICKUP_WEAPON 5
#define EVENT_PICKUP_ITEM 6
#define EVENT_FRAG_GAINED 7
#define EVENT_FRAG_LOST 8
#define EVENT_MONSTER_KILL 9
#define EVENT_DAMAGE_DEALT 10
#define EVENT_HIT_CONFIRMED 11
#define EVENT_SHOTS_FIRED 12
#define EVENT_GOAL_REACHED 13

static int extract_int(const char *line, const char *key, int fallback)
{
    const char *match = strstr(line, key);
    const char *colon;
    if (match == NULL) {
        return fallback;
    }
    colon = strchr(match, ':');
    if (colon == NULL) {
        return fallback;
    }
    return atoi(colon + 1);
}

static int extract_string(const char *line, const char *key, char *out, size_t out_size)
{
    const char *match = strstr(line, key);
    const char *colon;
    const char *cursor;
    size_t index = 0;
    if (match == NULL) {
        return 0;
    }
    colon = strchr(match, ':');
    if (colon == NULL) {
        return 0;
    }
    cursor = strchr(colon, '"');
    if (cursor == NULL) {
        return 0;
    }
    cursor += 1;
    while (*cursor && *cursor != '"' && index + 1 < out_size) {
        if (*cursor == '\\' && cursor[1]) {
            cursor += 1;
        }
        out[index++] = *cursor++;
    }
    if (*cursor != '"') {
        return 0;
    }
    out[index] = '\0';
    return 1;
}

static int parse_protocol_version_text(const char *value, int fallback)
{
    const char *cursor;
    int parsed;

    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    cursor = value;
    if (*cursor == 'v' || *cursor == 'V') {
        cursor += 1;
    }
    if (*cursor == '\0') {
        return fallback;
    }
    parsed = atoi(cursor);
    return parsed > 0 ? parsed : fallback;
}

static int current_region_id(int step);

typedef struct {
    const char *values[16];
    int count;
} binary_string_table_t;

static void write_bytes(const void *data, size_t size)
{
    if (size > 0) {
        fwrite(data, 1, size, stdout);
    }
}

static void write_u16_le(uint16_t value)
{
    unsigned char bytes[2];
    bytes[0] = (unsigned char)(value & 0xff);
    bytes[1] = (unsigned char)((value >> 8) & 0xff);
    write_bytes(bytes, sizeof(bytes));
}

static void write_i16_le(int value)
{
    write_u16_le((uint16_t)(int16_t)value);
}

static void write_u32_le(uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xff);
    bytes[1] = (unsigned char)((value >> 8) & 0xff);
    bytes[2] = (unsigned char)((value >> 16) & 0xff);
    bytes[3] = (unsigned char)((value >> 24) & 0xff);
    write_bytes(bytes, sizeof(bytes));
}

static void write_i32_le(int value)
{
    write_u32_le((uint32_t)(int32_t)value);
}

static void write_f32_le(float value)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    write_u32_le(bits.u);
}

static int binary_string_index(binary_string_table_t *table, const char *value)
{
    int i;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    for (i = 0; i < table->count; ++i) {
        if (strcmp(table->values[i], value) == 0) {
            return i + 1;
        }
    }
    if (table->count >= (int)(sizeof(table->values) / sizeof(table->values[0]))) {
        return 0;
    }
    table->values[table->count] = value;
    table->count += 1;
    return table->count;
}

static void write_binary_strings(const binary_string_table_t *table)
{
    int i;
    for (i = 0; i < table->count; ++i) {
        const char *value = table->values[i];
        size_t length = strlen(value);
        if (length > 0xffff) {
            length = 0xffff;
        }
        write_u16_le((uint16_t)length);
        write_bytes(value, length);
    }
}

static void write_action_binary(
    int move,
    int strafe,
    int look_yaw,
    int look_pitch,
    int fire,
    int jump,
    int weapon)
{
    write_i16_le(move);
    write_i16_le(strafe);
    write_i16_le(look_yaw);
    write_i16_le(look_pitch);
    write_i16_le(fire);
    write_i16_le(jump);
    write_i16_le(weapon);
}

static void write_binary_event(
    int event_code,
    int flags,
    int region_id,
    int delta,
    int weapon_id,
    int source_entity_num,
    int target_entity_num)
{
    write_u16_le((uint16_t)event_code);
    write_u16_le((uint16_t)flags);
    write_i32_le(region_id);
    write_i32_le(delta);
    write_i32_le(weapon_id);
    write_i32_le(source_entity_num);
    write_i32_le(target_entity_num);
}

static void write_step_binary(
    int step,
    int done,
    int look_yaw,
    int look_pitch,
    int fire,
    int jump,
    int weapon,
    double reward)
{
    binary_string_table_t strings = {0};
    int region_id = current_region_id(step);
    int previous_region_id = current_region_id(step > 0 ? step - 1 : 0);
    int visible_entity_num = reset_deathmatch ? 2 : 9;
    const char *visible_classname = reset_deathmatch ? "player" : "monster_ogre";
    int visible_count = done ? 0 : 1;
    int shots_fired = fire ? 1 : 0;
    int action_history_count = step > 0 ? 1 : 0;
    double x = (region_id - 1) * 256.0;
    double vx = (x - ((previous_region_id - 1) * 256.0)) / 5.0;
    int health = step >= 2 ? 85 : 100;
    int armor = step >= 3 ? 25 : 0;
    int ammo = fire ? 20 : 25;
    int damage_dealt = step == kill_event_step ? 40 : 0;
    int hit_count = step == kill_event_step ? 1 : 0;
    int took_damage = step == 2;
    int killed_target = step == kill_event_step;
    int picked_up_health = step == 1;
    int picked_up_weapon = fire && picked_up_health;
    int event_count =
        took_damage
        + (shots_fired > 0)
        + (damage_dealt > 0)
        + (hit_count > 0)
        + killed_target
        + picked_up_health
        + picked_up_weapon
        + done;
    uint16_t flags = 0;

    if (done) {
        flags |= STEP_FLAG_DONE | STEP_FLAG_GOAL_REACHED;
    }
    if (visible_count > 0) {
        binary_string_index(&strings, visible_classname);
    }

    write_bytes(STEP_BINARY_MAGIC, 4);
    write_u16_le(STEP_BINARY_VERSION);
    write_u16_le(flags);
    write_f32_le((float)reward);
    write_i32_le(step);
    write_i32_le(step);
    write_i32_le(region_id);
    write_i32_le(current_frags);
    write_i32_le(current_monster_kills);
    write_i32_le(4);
    write_i32_le(health);
    write_i32_le(armor);
    write_i32_le(ammo);
    write_i32_le(fire ? 3 : 1);
    write_i32_le(0);
    write_i32_le(0);
    write_i32_le(0);
    write_i32_le(0);
    write_i32_le(0);
    write_i32_le(1);
    write_f32_le(0.0f);
    write_f32_le((float)x);
    write_f32_le(0.0f);
    write_f32_le(0.0f);
    write_f32_le((float)vx);
    write_f32_le(0.0f);
    write_f32_le(0.0f);
    write_f32_le(0.0f);
    write_f32_le(0.0f);
    write_f32_le(0.0f);
    write_i32_le(damage_dealt);
    write_i32_le(total_damage_dealt);
    write_i32_le(fire ? 3 : 0);
    write_i32_le(hit_count);
    write_i32_le(total_hit_count);
    write_i32_le(shots_fired);
    write_i32_le(total_shots_fired);
    write_i32_le(done ? DONE_REASON_GOAL_REACHED : DONE_REASON_NONE);
    for (int i = 0; i < 9; ++i) {
        write_i32_le(weapon_damage_dealt[i]);
    }
    for (int i = 0; i < 9; ++i) {
        write_i32_le(weapon_hits_landed[i]);
    }
    for (int i = 0; i < 9; ++i) {
        write_i32_le(weapon_shots_fired[i]);
    }
    write_action_binary(1, 0, look_yaw, look_pitch, fire, jump, weapon);
    write_u16_le((uint16_t)action_history_count);
    write_u16_le((uint16_t)visible_count);
    write_u16_le((uint16_t)event_count);
    write_u16_le(0);
    write_u16_le((uint16_t)strings.count);

    if (action_history_count > 0) {
        write_action_binary(1, 0, previous_look_yaw, previous_look_pitch, previous_fire, previous_jump, previous_weapon);
    }

    if (visible_count > 0) {
        write_i32_le(visible_entity_num);
        write_i32_le(visible_entity_num);
        write_i32_le(3);
        write_u16_le((uint16_t)binary_string_index(&strings, visible_classname));
        write_u16_le(0);
        write_u16_le(0);
        write_f32_le(512.0f);
        write_f32_le(reset_deathmatch ? 32.0f : -32.0f);
        write_f32_le(0.0f);
    }

    if (took_damage) {
        write_binary_event(EVENT_DAMAGE_TAKEN, EVENT_FLAG_HAS_DELTA, 3, 15, 0, 0, 0);
    }
    if (shots_fired > 0) {
        write_binary_event(EVENT_SHOTS_FIRED, EVENT_FLAG_HAS_DELTA | EVENT_FLAG_HAS_WEAPON_ID, region_id, 1, 3, 1, 0);
    }
    if (damage_dealt > 0) {
        write_binary_event(EVENT_DAMAGE_DEALT, EVENT_FLAG_HAS_DELTA | EVENT_FLAG_HAS_WEAPON_ID, 3, damage_dealt, 3, 1, visible_entity_num);
    }
    if (hit_count > 0) {
        write_binary_event(EVENT_HIT_CONFIRMED, EVENT_FLAG_HAS_DELTA | EVENT_FLAG_HAS_WEAPON_ID, 3, hit_count, 3, 1, visible_entity_num);
    }
    if (killed_target) {
        write_binary_event(reset_deathmatch ? EVENT_FRAG_GAINED : EVENT_MONSTER_KILL, EVENT_FLAG_HAS_DELTA, 3, 1, 0, 0, 0);
    }
    if (picked_up_health) {
        write_binary_event(EVENT_PICKUP_HEALTH, EVENT_FLAG_HAS_DELTA, 2, 25, 0, 0, 0);
    }
    if (picked_up_weapon) {
        write_binary_event(EVENT_PICKUP_WEAPON, EVENT_FLAG_HAS_WEAPON_ID, 2, 0, 3, 0, 0);
    }
    if (done) {
        write_binary_event(EVENT_GOAL_REACHED, 0, 4, 0, 0, 0, 0);
    }

    write_binary_strings(&strings);
    fflush(stdout);
}

static int current_region_id(int step)
{
    if (step <= 0) {
        return 1;
    }
    if (step == 1) {
        return 2;
    }
    if (step == 2) {
        return 3;
    }
    return 4;
}

static void write_map_state(void)
{
    printf("{\"goal_region_ids\":[4],\"map_id\":\"%s\",\"metadata\":{\"distance_to_goal\":{\"1\":3.0,\"2\":2.0,\"3\":1.0,\"4\":0.0},\"max_distance_to_goal\":3.0,\"source\":\"native-stub\"},\"regions\":[", map_id);
    printf("{\"bounds_max\":[128.0,128.0,64.0],\"bounds_min\":[-128.0,-128.0,-64.0],\"center\":[0.0,0.0,0.0],\"neighbors\":[2],\"object_ids\":[\"spawn_0000\"],\"region_id\":1,\"visibility_hints\":[2]},");
    printf("{\"bounds_max\":[384.0,128.0,64.0],\"bounds_min\":[128.0,-128.0,-64.0],\"center\":[256.0,0.0,0.0],\"neighbors\":[1,3],\"object_ids\":[\"item_0001\"],\"region_id\":2,\"visibility_hints\":[1,3]},");
    printf("{\"bounds_max\":[640.0,128.0,64.0],\"bounds_min\":[384.0,-128.0,-64.0],\"center\":[512.0,0.0,0.0],\"neighbors\":[2,4],\"object_ids\":[],\"region_id\":3,\"visibility_hints\":[2,4]},");
    printf("{\"bounds_max\":[896.0,128.0,64.0],\"bounds_min\":[640.0,-128.0,-64.0],\"center\":[768.0,0.0,0.0],\"neighbors\":[3],\"object_ids\":[\"goal_0002\"],\"region_id\":4,\"visibility_hints\":[3]}],");
    printf("\"spawn_region_ids\":[1],\"static_objects\":[");
    printf("{\"angles\":[0.0,0.0,0.0],\"category\":\"spawn\",\"classname\":\"info_player_start\",\"object_id\":\"spawn_0000\",\"origin\":[0.0,0.0,0.0],\"properties\":{},\"region_id\":1},");
    printf("{\"angles\":[0.0,0.0,0.0],\"category\":\"item\",\"classname\":\"item_health\",\"object_id\":\"item_0001\",\"origin\":[256.0,0.0,0.0],\"properties\":{},\"region_id\":2},");
    printf("{\"angles\":[0.0,0.0,0.0],\"category\":\"goal\",\"classname\":\"trigger_changelevel\",\"object_id\":\"goal_0002\",\"origin\":[768.0,0.0,0.0],\"properties\":{},\"region_id\":4}]}");
}

static void write_visible_entities(int done)
{
    if (done) {
        printf("[]");
        return;
    }
    putchar('[');
    if (reset_deathmatch) {
        printf("{\"angles\":[0.0,180.0,0.0],\"classname\":\"player\",\"entity_id\":\"entity_0002\",\"entity_num\":2,\"frame\":0,\"model_id\":12,\"origin\":[512.0,32.0,0.0],\"properties\":{\"effects\":0,\"frags\":0,\"health\":60},\"region_id\":3,\"velocity\":[0.0,0.0,0.0],\"visible\":true}");
    } else {
        printf("{\"angles\":[0.0,180.0,0.0],\"classname\":\"monster_ogre\",\"entity_id\":\"entity_0009\",\"entity_num\":9,\"frame\":0,\"model_id\":24,\"origin\":[512.0,-32.0,0.0],\"properties\":{\"effects\":0,\"frags\":0,\"health\":60},\"region_id\":3,\"velocity\":[0.0,0.0,0.0],\"visible\":true}");
    }
    putchar(']');
}

static void write_events(int step, int fire, int done)
{
    int wrote = 0;
    int damage_dealt = step == kill_event_step ? 40 : 0;
    int hit_count = step == kill_event_step ? 1 : 0;
    int shots_fired = fire ? 1 : 0;
    putchar('[');
    if (step == 2) {
        printf("{\"event_type\":\"damage_taken\",\"payload\":{\"delta\":15},\"region_id\":3,\"source_id\":\"\",\"target_id\":\"\"}");
        wrote = 1;
    }
    if (shots_fired > 0) {
        if (wrote) {
            putchar(',');
        }
        printf("{\"event_type\":\"shots_fired\",\"payload\":{\"delta\":1,\"weapon_id\":3},\"region_id\":%d,\"source_id\":\"entity_0001\",\"target_id\":\"\"}", current_region_id(step));
        wrote = 1;
    }
    if (damage_dealt > 0) {
        if (wrote) {
            putchar(',');
        }
        printf("{\"event_type\":\"damage_dealt\",\"payload\":{\"delta\":%d,\"weapon_id\":3},\"region_id\":3,\"source_id\":\"entity_0001\",\"target_id\":\"entity_0002\"}", damage_dealt);
        wrote = 1;
    }
    if (hit_count > 0) {
        if (wrote) {
            putchar(',');
        }
        printf("{\"event_type\":\"hit_confirmed\",\"payload\":{\"delta\":%d,\"weapon_id\":3},\"region_id\":3,\"source_id\":\"entity_0001\",\"target_id\":\"entity_0002\"}", hit_count);
        wrote = 1;
    }
    if (step == kill_event_step) {
        if (wrote) {
            putchar(',');
        }
        if (reset_deathmatch) {
            printf("{\"event_type\":\"frag_gained\",\"payload\":{\"delta\":1},\"region_id\":3,\"source_id\":\"\",\"target_id\":\"\"}");
        } else {
            printf("{\"event_type\":\"monster_kill\",\"payload\":{\"delta\":1},\"region_id\":3,\"source_id\":\"\",\"target_id\":\"\"}");
        }
        wrote = 1;
    }
    if (step == 1) {
        if (wrote) {
            putchar(',');
        }
        printf("{\"event_type\":\"pickup_health\",\"payload\":{\"delta\":25},\"region_id\":2,\"source_id\":\"\",\"target_id\":\"\"}");
        wrote = 1;
    }
    if (fire && step == 1) {
        if (wrote) {
            putchar(',');
        }
        printf("{\"event_type\":\"pickup_weapon\",\"payload\":{\"weapon_id\":3},\"region_id\":2,\"source_id\":\"\",\"target_id\":\"\"}");
        wrote = 1;
    }
    if (done) {
        if (wrote) {
            putchar(',');
        }
        printf("{\"event_type\":\"goal_reached\",\"payload\":{},\"region_id\":4,\"source_id\":\"\",\"target_id\":\"\"}");
    }
    putchar(']');
}

static void write_world_tick(
    int step,
    int done,
    int look_yaw,
    int look_pitch,
    int fire,
    int jump,
    int weapon)
{
    int region_id = current_region_id(step);
    int previous_region_id = current_region_id(step > 0 ? step - 1 : 0);
    double x = (region_id - 1) * 256.0;
    int health = step >= 2 ? 85 : 100;
    int armor = step >= 3 ? 25 : 0;
    int ammo = fire ? 20 : 25;
    int damage_dealt = step == kill_event_step ? 40 : 0;
    int hit_count = step == kill_event_step ? 1 : 0;
    int shots_fired = fire ? 1 : 0;
    int damage_weapon_id = fire ? 3 : 0;

    printf("{\"action_history\":[");
    if (step > 0) {
        printf(
            "{\"move\":1,\"strafe\":0,\"look_yaw\":%d,\"look_pitch\":%d,\"fire\":%d,\"jump\":%d,\"weapon\":%d}",
            previous_look_yaw,
            previous_look_pitch,
            previous_fire,
            previous_jump,
            previous_weapon);
    }
    printf("],\"action_label\":{\"move\":1,\"strafe\":0,\"look_yaw\":%d,\"look_pitch\":%d,\"fire\":%d,\"jump\":%d,\"weapon\":%d},\"current_region_id\":%d,\"debug\":{\"damage_dealt\":%d,\"damage_dealt_total\":%d,\"damage_weapon_id\":%d,\"frags\":%d,\"hit_count\":%d,\"hit_count_total\":%d,\"monster_kills\":%d,\"monster_total\":4,\"seed\":%d,\"shots_fired\":%d,\"shots_fired_total\":%d,\"weapon_damage_dealt_total\":[%d,%d,%d,%d,%d,%d,%d,%d,%d],\"weapon_hits_landed_total\":[%d,%d,%d,%d,%d,%d,%d,%d,%d],\"weapon_shots_fired_total\":[%d,%d,%d,%d,%d,%d,%d,%d,%d]},\"done\":%s,\"done_reason\":\"%s\",\"episode_id\":\"stub-episode\",\"events\":",
           look_yaw,
           look_pitch,
           fire,
           jump,
           weapon,
           region_id,
           damage_dealt,
           total_damage_dealt,
           damage_weapon_id,
           current_frags,
           hit_count,
           total_hit_count,
           current_monster_kills,
           episode_seed,
           shots_fired,
           total_shots_fired,
           weapon_damage_dealt[0], weapon_damage_dealt[1], weapon_damage_dealt[2], weapon_damage_dealt[3], weapon_damage_dealt[4], weapon_damage_dealt[5], weapon_damage_dealt[6], weapon_damage_dealt[7], weapon_damage_dealt[8],
           weapon_hits_landed[0], weapon_hits_landed[1], weapon_hits_landed[2], weapon_hits_landed[3], weapon_hits_landed[4], weapon_hits_landed[5], weapon_hits_landed[6], weapon_hits_landed[7], weapon_hits_landed[8],
           weapon_shots_fired[0], weapon_shots_fired[1], weapon_shots_fired[2], weapon_shots_fired[3], weapon_shots_fired[4], weapon_shots_fired[5], weapon_shots_fired[6], weapon_shots_fired[7], weapon_shots_fired[8],
           done ? "true" : "false",
           done ? "goal_reached" : "");
    write_events(step, fire, done);
    printf(",\"map_id\":\"%s\",\"player\":{\"ammo\":%d,\"armor\":%d,\"grounded\":true,\"health\":%d,\"origin\":[%.1f,0.0,0.0],\"velocity\":[%.1f,0.0,0.0],\"view_angles\":[0.0,0.0,0.0],\"weapon_id\":%d},\"reset\":%s,\"tick\":%d,\"visible_entities\":",
           map_id,
           ammo,
           armor,
           health,
           x,
           (x - ((previous_region_id - 1) * 256.0)) / 5.0,
           fire ? 3 : 1,
           step == 0 ? "true" : "false",
           step);
    write_visible_entities(done);
    printf("}");
}

int main(int argc, char **argv)
{
    char line[4096];
    (void)argc;
    (void)argv;

    while (fgets(line, sizeof(line), stdin) != NULL) {
        if (strstr(line, "\"op\"") != NULL && strstr(line, "hello") != NULL) {
            const char *map_match = strstr(line, "\"map_id\"");
            const char *quote;
            char version_text[16];
            char step_format[32];
            int requested_protocol_version;
            size_t len = 0;
            if (map_match != NULL) {
                quote = strchr(map_match + 8, '"');
                if (quote != NULL) {
                    const char *value = quote + 1;
                    const char *end = strchr(value, '"');
                    if (end != NULL) {
                        len = (size_t)(end - value);
                        if (len >= sizeof(map_id)) {
                            len = sizeof(map_id) - 1;
                        }
                        memcpy(map_id, value, len);
                        map_id[len] = '\0';
                    }
                }
            }
            requested_protocol_version = 3;
            version_text[0] = '\0';
            step_format[0] = '\0';
            if (extract_string(line, "\"protocol_version\"", version_text, sizeof(version_text))) {
                requested_protocol_version = parse_protocol_version_text(version_text, 3);
            }
            binary_step_enabled = extract_string(line, "\"step_format\"", step_format, sizeof(step_format))
                && strcmp(step_format, "binary_v1") == 0
                && requested_protocol_version >= 3;
            tick_hz = extract_int(line, "\"tick_hz\"", tick_hz);
            printf("{\"capabilities\":[\"binary_step_v1\",\"reset_options\",\"world_tick_only\"],\"map_id\":\"%s\",\"map_state\":", map_id);
            write_map_state();
            printf(",\"ok\":true,\"protocol_version\":\"v3\",\"server\":\"native-stub\",\"tick_hz\":%d}\n", tick_hz);
            fflush(stdout);
            continue;
        }

        if (strstr(line, "\"op\"") != NULL && strstr(line, "reset") != NULL) {
            step_count = 0;
            episode_seed = extract_int(line, "\"seed\"", -1);
            previous_look_yaw = 12;
            previous_look_pitch = 12;
            previous_fire = 0;
            previous_jump = 0;
            previous_weapon = 0;
            reset_maxplayers = extract_int(line, "\"maxplayers\"", 1);
            reset_deathmatch = extract_int(line, "\"deathmatch\"", 0);
            reset_teamplay = extract_int(line, "\"teamplay\"", 0);
            current_frags = 0;
            current_monster_kills = 0;
            kill_event_step = -1;
            total_damage_dealt = 0;
            total_hit_count = 0;
            total_shots_fired = 0;
            memset(weapon_damage_dealt, 0, sizeof(weapon_damage_dealt));
            memset(weapon_hits_landed, 0, sizeof(weapon_hits_landed));
            memset(weapon_shots_fired, 0, sizeof(weapon_shots_fired));
            printf("{\"info\":{\"deathmatch\":%d,\"map_id\":\"%s\",\"maxplayers\":%d,\"seed\":%d,\"teamplay\":%d},\"ok\":true,\"world_tick\":",
                   reset_deathmatch,
                   map_id,
                   reset_maxplayers,
                   episode_seed,
                   reset_teamplay);
            write_world_tick(step_count, 0, 12, 12, 0, 0, 0);
            printf("}\n");
            fflush(stdout);
            continue;
        }

        if (strstr(line, "\"op\"") != NULL && strstr(line, "step") != NULL) {
            int look_yaw = extract_int(line, "\"look_yaw\"", 12);
            int look_pitch = extract_int(line, "\"look_pitch\"", 12);
            int fire = extract_int(line, "\"fire\"", 0);
            int jump = extract_int(line, "\"jump\"", 0);
            int weapon = extract_int(line, "\"weapon\"", 0);
            int done;
            double reward;

            step_count += 1;
            if (fire && kill_event_step < 0) {
                kill_event_step = step_count;
                if (reset_deathmatch) {
                    current_frags = 1;
                } else {
                    current_monster_kills = 1;
                }
                total_damage_dealt += 40;
                total_hit_count += 1;
                weapon_damage_dealt[3] += 40;
                weapon_hits_landed[3] += 1;
            }
            if (fire) {
                total_shots_fired += 1;
                weapon_shots_fired[3] += 1;
            }
            done = step_count >= 4;
            reward = done ? 1.0 : 0.1;
            if (binary_step_enabled) {
                write_step_binary(step_count, done, look_yaw, look_pitch, fire, jump, weapon, reward);
            } else {
                printf("{\"done\":%s,\"info\":{\"deathmatch\":%d,\"goal_reached\":%s,\"maxplayers\":%d,\"steps\":%d,\"teamplay\":%d},\"ok\":true,\"reward\":%.3f,\"world_tick\":",
                       done ? "true" : "false",
                       reset_deathmatch,
                       done ? "true" : "false",
                       reset_maxplayers,
                       step_count,
                       reset_teamplay,
                       reward);
                write_world_tick(step_count, done, look_yaw, look_pitch, fire, jump, weapon);
                printf("}\n");
                fflush(stdout);
            }
            previous_look_yaw = look_yaw;
            previous_look_pitch = look_pitch;
            previous_fire = fire;
            previous_jump = jump;
            previous_weapon = weapon;
            continue;
        }

        if (strstr(line, "\"op\"") != NULL && strstr(line, "shutdown") != NULL) {
            printf("{\"ok\":true}\n");
            fflush(stdout);
            return 0;
        }

        printf("{\"error\":\"unsupported op\",\"ok\":false}\n");
        fflush(stdout);
    }

    return 0;
}
