#include <stdio.h>
#include <stdlib.h>
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
static int damage_event_step = -1;
static int total_damage_dealt = 0;
static int total_hit_count = 0;
static int total_shots_fired = 0;
static int weapon_damage_dealt[9] = {0};
static int weapon_hits_landed[9] = {0};
static int weapon_shots_fired[9] = {0};

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

static void write_json_string(const char *value)
{
    const char *cursor;
    putchar('"');
    for (cursor = value; *cursor; ++cursor) {
        if (*cursor == '"' || *cursor == '\\') {
            putchar('\\');
        }
        putchar(*cursor);
    }
    putchar('"');
}

static void write_obs(int step)
{
    int i;
    putchar('[');
    for (i = 0; i < 20; ++i) {
        double value = 0.0;
        if (i == 0) {
            value = (double)step / 10.0;
        } else if (i == 12) {
            value = step >= 3 ? 1.0 : 0.0;
        }
        if (i > 0) {
            putchar(',');
        }
        printf("%.3f", value);
    }
    putchar(']');
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
            tick_hz = extract_int(line, "\"tick_hz\"", tick_hz);
            printf("{\"capabilities\":[\"legacy_obs\",\"world_v2\",\"reset_options\"],\"map_id\":\"%s\",\"map_state\":", map_id);
            write_map_state();
            printf(",\"ok\":true,\"protocol_version\":\"v2\",\"server\":\"native-stub\",\"tick_hz\":%d}\n", tick_hz);
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
            damage_event_step = -1;
            total_damage_dealt = 0;
            total_hit_count = 0;
            total_shots_fired = 0;
            memset(weapon_damage_dealt, 0, sizeof(weapon_damage_dealt));
            memset(weapon_hits_landed, 0, sizeof(weapon_hits_landed));
            memset(weapon_shots_fired, 0, sizeof(weapon_shots_fired));
            printf("{\"info\":{\"deathmatch\":%d,\"map_id\":\"%s\",\"maxplayers\":%d,\"seed\":%d,\"teamplay\":%d},\"obs\":",
                   reset_deathmatch,
                   map_id,
                   reset_maxplayers,
                   episode_seed,
                   reset_teamplay);
            write_obs(step_count);
            printf(",\"ok\":true,\"world_tick\":");
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
            if (step_count == 2) {
                damage_event_step = step_count;
            }
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
            printf("{\"done\":%s,\"info\":{\"deathmatch\":%d,\"goal_reached\":%s,\"maxplayers\":%d,\"steps\":%d,\"teamplay\":%d},\"obs\":",
                   done ? "true" : "false",
                   reset_deathmatch,
                   done ? "true" : "false",
                   reset_maxplayers,
                   step_count,
                   reset_teamplay);
            write_obs(step_count);
            printf(",\"ok\":true,\"reward\":%.3f,\"world_tick\":", reward);
            write_world_tick(step_count, done, look_yaw, look_pitch, fire, jump, weapon);
            printf("}\n");
            previous_look_yaw = look_yaw;
            previous_look_pitch = look_pitch;
            previous_fire = fire;
            previous_jump = jump;
            previous_weapon = weapon;
            fflush(stdout);
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
