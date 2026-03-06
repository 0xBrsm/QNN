#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int step_count = 0;
static int episode_seed = -1;
static char map_id[64] = "UNKNOWN";
static int tick_hz = 20;

static int extract_int(const char *line, const char *key, int fallback)
{
    const char *match = strstr(line, key);
    if (match == NULL) {
        return fallback;
    }
    match = strchr(match, ':');
    if (match == NULL) {
        return fallback;
    }
    return atoi(match + 1);
}

static void write_obs(int step)
{
    int i;
    putchar('[');
    for (i = 0; i < 20; ++i) {
        double value = (i == 0) ? (double)step / 10.0 : ((i == 12) ? (step >= 3 ? 1.0 : 0.0) : 0.0);
        if (i > 0) {
            putchar(',');
        }
        printf("%.3f", value);
    }
    putchar(']');
}

int main(void)
{
    char line[4096];
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
            printf("{\"ok\":true,\"map_id\":\"%s\",\"server\":\"native-stub\",\"tick_hz\":%d}\n", map_id, tick_hz);
            fflush(stdout);
            continue;
        }

        if (strstr(line, "\"op\"") != NULL && strstr(line, "reset") != NULL) {
            step_count = 0;
            episode_seed = extract_int(line, "\"seed\"", -1);
            printf("{\"info\":{\"map_id\":\"%s\",\"seed\":%d},\"obs\":", map_id, episode_seed);
            write_obs(step_count);
            printf(",\"ok\":true}\n");
            fflush(stdout);
            continue;
        }

        if (strstr(line, "\"op\"") != NULL && strstr(line, "step") != NULL) {
            int used = strstr(line, "\"use\":1") != NULL;
            int done;
            double reward;
            step_count += 1;
            done = used || step_count >= 4;
            reward = done ? 1.0 : 0.1;
            printf("{\"done\":%s,\"info\":{\"goal_reached\":%s,\"steps\":%d},\"obs\":",
                   done ? "true" : "false",
                   done ? "true" : "false",
                   step_count);
            write_obs(step_count);
            printf(",\"ok\":true,\"reward\":%.3f}\n", reward);
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
