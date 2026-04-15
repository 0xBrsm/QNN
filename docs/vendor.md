# Vendored Dependencies

This repo treats `vendor/` as local bootstrap state, not tracked source. Git
ignores the entire tree. The source of truth for third-party inputs is this
document.

## Required Inputs

| Dependency | Local path | Upstream | Version / commit | Used by |
| --- | --- | --- | --- | --- |
| CleanFixedQuakeC | `vendor/quakec` | <https://github.com/Jason2Brownlee/CleanFixedQuakeC.git> | `dacc69ef13d2c961776b708f33ad2be8e87da224` | `scripts/build-progs.sh` base QuakeC sources |
| FrikBotNex | `vendor/frikbotnex` | <https://github.com/0xBrsm/FrikBotNex.git> | `c767dbf325950b62a13ba22c68a395b39d6ad376` | `scripts/build-progs.sh` bot logic and waypoints |
| RecastNavigation | `vendor/recastnavigation` | <https://github.com/recastnavigation/recastnavigation> | `v1.6.0` / `6dc1667f580357e8a2154c28b7867bea7e8ad3a7` | `engine/build/build_common.sh` |
| ericw-tools | `vendor/ericw-tools/bin` or `PATH` | <https://github.com/ericwa/ericw-tools> | `0.18.1` binary tool bundle | `mapgen/compile.py` |

## Fetched During Install

| Dependency | Source | Version / commit | Used by |
| --- | --- | --- | --- |
| Quake-Tools | <https://github.com/id-Software/Quake-Tools.git> | `c0d1b91c74eb654365ac7755bc837e497caaca73` | `scripts/install_frikbotnex.py`, `scripts/install_training_gamedir.py` for `qcc` |

## Bootstrap Notes

- `vendor/quakec` needs only `LICENSE.txt` plus the upstream `qc/` tree.
- `vendor/frikbotnex` needs only the upstream `src/frikbot/` and
  `src/waypoints/` trees.
- `vendor/recastnavigation` needs only `Recast/{Include,Source}`,
  `Detour/{Include,Source}`, and `License.txt`.
- `vendor/ericw-tools` is optional if the tools are already installed on
  `PATH`.
- Local QNN-specific changes live outside `vendor/`. For QuakeC, the repo's
  `frikbotnex/` overlays are applied after the vendored upstream inputs are
  staged.
- `vendor/quake-src` may exist locally for reference, but it is not required
  by the current build.

## Local Asset Outputs

- `assets/frikbotnex_train/progs.dat` is a generated output. Rebuild it with
  `scripts/build-progs.sh` after staging `vendor/quakec` and
  `vendor/frikbotnex`.
- `assets/id1/maps/arena_box.bsp` is currently treated as a local asset. The
  repo does not yet carry a canonical tracked `.map` source for it, so a clean
  clone cannot recreate that BSP from repo state alone today.
