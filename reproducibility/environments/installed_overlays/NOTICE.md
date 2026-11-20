# Installed Python source snapshots

These files preserve the source environment's installed Python code as of
November 20, 2026. The manifest records each original wheel RECORD hash and the
installed replacement hash. They are copied only into matching package versions.

| Files | Upstream project / version | License |
| --- | --- | --- |
| `augmax/` | https://github.com/khdlr/augmax / 0.4.1 | Apache-2.0; copyright Konrad Heidler retained in source |
| `mplib/planner.py` | https://github.com/haosulab/MPlib / 0.2.1 | MIT; see `licenses/mplib-LICENSE` |
| `sapien/_oidn_tricks.py`, `sapien/wrapper/urdf_loader.py` | https://github.com/haosulab/SAPIEN / 3.0.1 | Apache-2.0 notices retained; see `licenses/sapien-LICENSE` |

`licenses/Apache-2.0.txt` contains the full Apache license text. The SAPIEN loader
selects OIDN 2.3.3; the URDF and MPLIB changes originated in the pinned RLInf
RoboTwin installation procedure. The Augmax installed files differ from their
wheel RECORD values; their exact installed contents are retained rather than
assuming an unverified clean-wheel replacement is equivalent. No claim of
authorship of these upstream projects is made.
