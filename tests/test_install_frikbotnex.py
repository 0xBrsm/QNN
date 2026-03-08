from __future__ import annotations

from quake_ai.install_frikbotnex import _merge_progs_src, _patch_client_qc, _patch_defs_qc, _patch_world_qc


def test_merge_progs_src_inserts_waypoints_and_bot_sources_after_defs() -> None:
    source = "../progs.dat\n\ndefs.qc\nsubs.qc\nworld.qc\n"

    merged = _merge_progs_src(source)

    assert merged.startswith("progs.dat\n")
    assert "defs.qc\nwaypoints/map_dm1.qc\n" in merged
    assert "waypoints/map_dm6.qc\nfrikbot/bot.qc\n" in merged
    assert merged.rstrip().endswith("world.qc")


def test_patch_defs_qc_comments_builtin_overrides() -> None:
    source = "\n".join(
        [
            "void(entity e, float chan, string samp, float vol, float atten) sound = #8;",
            "void(entity client, string s)stuffcmd = #21;",
            "void(entity client, string s) sprint = #24;",
            "vector(entity e, float speed) aim = #44;\t\t// returns the shooting vector",
            "void(float to, float f) WriteByte\t\t= #52;",
            "void(float to, float f) WriteChar\t\t= #53;",
            "void(float to, float f) WriteShort\t\t= #54;",
            "void(float to, float f) WriteLong\t\t= #55;",
            "void(float to, float f) WriteCoord\t\t= #56;",
            "void(float to, float f) WriteAngle\t\t= #57;",
            "void(float to, string s) WriteString\t= #58;",
            "void(float to, entity s) WriteEntity\t= #59;",
            "void(entity client, string s) centerprint = #73;\t// sprint, but in middle",
            "void(entity e) setspawnparms\t\t= #78;\t\t// set parm1... to the",
        ]
    )

    patched = _patch_defs_qc(source)

    assert patched.count("// FrikBot override:") == 14
    assert "FrikBot override: void(entity client, string s)stuffcmd = #21;" in patched


def test_patch_world_and_client_hooks_insert_frikbot_callbacks() -> None:
    world_source = 'void() worldspawn =\n{\n\tlastspawn = world;\n\tInitBodyQue ();\n}\n\nvoid() StartFrame =\n{\n\tteamplay = cvar("teamplay");\n}\n'
    client_source = "void() PlayerPreThink =\n{\n}\n\nvoid() PlayerPostThink =\n{\n}\n\nvoid() ClientConnect =\n{\n}\n\nvoid() ClientDisconnect =\n{\n}\n"

    world_patched = _patch_world_qc(world_source)
    client_patched = _patch_client_qc(client_source)

    assert "BotInit();\t// FrikBot" in world_patched
    assert "BotFrame();\t// FrikBot" in world_patched
    assert "if (BotPreFrame())" in client_patched
    assert "if (BotPostFrame())" in client_patched
    assert "ClientInRankings();\t// FrikBot" in client_patched
    assert "ClientDisconnected();\t// FrikBot" in client_patched
