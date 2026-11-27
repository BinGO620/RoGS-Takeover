"""an internal note 探针 B 门禁：MapLoad 默认关闭 + fail-fast + 记账。

对齐 `the pre-registered protocol.md` §3.1/§3.3：
1. **关闭态**：`MapLoad` 缺席/enabled=false ⇒ `mapload_enabled` 恒 False，
   且 `mapload_from_ply` **不被调用**（结构守卫）；
2. **fail-fast**：enabled=true 但 `ply_path` 为 null ⇒ 显式 `ValueError`，**不静默跳过**
   （否则探针臂会静默退化成自建图，读数全是假的）；
3. **文件缺失**：路径不存在 ⇒ `FileNotFoundError`；
4. **记账**：`mapload_summary.json` 的 `enabled/ply_path/n_prim/translate_m/kf_index` 齐全；
5. **源码接线守卫**：生效点在 `initialize_map` 之前，且关闭态走原路径。

用法：/data/conda_envs/monogs-ours/bin/python -m pytest tests/mapload -v
"""
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.slam_backend import mapload_cfg, mapload_enabled  # noqa: E402


# ── 1. 关闭态 ────────────────────────────────────────────────────────────
def test_missing_section_defaults_off():
    assert mapload_enabled({}) is False
    assert mapload_cfg({})["enabled"] is False


def test_explicit_false():
    cfg = {"MapLoad": {"enabled": False, "ply_path": "/tmp/x.ply"}}
    assert mapload_enabled(cfg) is False


def test_enabled_true():
    cfg = {"MapLoad": {"enabled": True, "ply_path": "/tmp/x.ply", "translate_m": 0.3}}
    assert mapload_enabled(cfg) is True
    c = mapload_cfg(cfg)
    assert c["ply_path"] == "/tmp/x.ply"
    assert c["translate_m"] == pytest.approx(0.3)


def test_translate_default_zero():
    c = mapload_cfg({"MapLoad": {"enabled": True, "ply_path": "/tmp/x.ply"}})
    assert c["translate_m"] == 0.0


# ── 2. fail-fast（关键：防探针臂静默退化）──────────────────────────────────
def test_fail_fast_when_ply_path_null(tmp_path):
    """enabled=true 但没给 ply_path ⇒ ValueError，不是静默跳过。"""
    from utils.slam_backend import BackEnd

    be = object.__new__(BackEnd)          # 不跑 mp.Process.__init__
    be.config = {
        "MapLoad": {"enabled": True, "ply_path": None},
        "Results": {"save_dir": str(tmp_path)},
    }
    be.gaussians = types.SimpleNamespace()
    be.opt_params = None
    with pytest.raises(ValueError, match="ply_path is null"):
        BackEnd.mapload_from_ply(be, 0, None)


def test_fail_fast_when_file_missing(tmp_path):
    from utils.slam_backend import BackEnd

    be = object.__new__(BackEnd)
    be.config = {
        "MapLoad": {"enabled": True, "ply_path": str(tmp_path / "nope.ply")},
        "Results": {"save_dir": str(tmp_path)},
    }
    be.gaussians = types.SimpleNamespace()
    be.opt_params = None
    with pytest.raises(FileNotFoundError, match="not found"):
        BackEnd.mapload_from_ply(be, 0, None)


# ── 3. 记账（DISCIPLINE #9：静默组件须有可观测读数）────────────────────────
def test_summary_written_with_all_keys(tmp_path):
    import json

    from utils.slam_backend import BackEnd

    ply = tmp_path / "m.ply"
    ply.write_bytes(b"ply\n")             # 内容无关：load_ply 被桩替换

    class _G:
        def __init__(self):
            import torch
            self._xyz = torch.zeros((7, 3))
            self.optimizer = object()      # 旧缺陷场景：已有优化器
            self.unique_kfIDs = None
            self.setup_args = None

        @property
        def get_xyz(self):
            return self._xyz

        def load_ply(self, path, kf_anchor=None):
            # 契约（the pre-registered protocol §1）：载入即一等公民——锚定 KF 并入
            # load_ply 本体，n_obs/lineage 按新基元语义初始化。
            import torch
            self.loaded = path
            self.kf_anchor = kf_anchor
            self.unique_kfIDs = torch.full((7,), int(kf_anchor), dtype=torch.int32)

        def training_setup(self, args):
            # 新契约事实：training_setup **无条件**重建（旧实现以 optimizer is None
            # 为门 ⇒ slam.py:297 的 fork 前优化器使门恒假 ⇒ 优化器持初始图旧张量，
            # W1 复现钉死的 10917 vs 25503 脱钩即此）。
            self.setup_args = args
            self.optimizer = object()

        def contract_repair_summary(self):
            return {"static_extend": 0, "static_reset": 0, "lineage_rebuild": 0}

    be = object.__new__(BackEnd)
    be.config = {
        "MapLoad": {"enabled": True, "ply_path": str(ply), "translate_m": 0.3},
        "Results": {"save_dir": str(tmp_path)},
    }
    be.gaussians = _G()
    be.opt_params = {"training": "args-sentinel"}
    be.occ_aware_visibility = {}          # 真实 BackEnd 在 __init__ 里置 {}，桩须同构
    BackEnd.mapload_from_ply(be, 42, None)

    summary = json.loads((tmp_path / "mapload_summary.json").read_text())
    assert summary["enabled"] is True
    assert summary["ply_path"] == str(ply)
    assert summary["n_prim"] == 7
    assert summary["translate_m"] == pytest.approx(0.3)
    assert summary["kf_index"] == 42
    # 契约新事实：优化器无条件重建 + 修复计数器暴露 + 锚定 KF 并入 load_ply
    assert summary["optimizer_rebuilt"] is True
    assert summary["contract_repair"] == {
        "static_extend": 0, "static_reset": 0, "lineage_rebuild": 0
    }
    assert be.gaussians.kf_anchor == 42
    assert int(be.gaussians.unique_kfIDs[0]) == 42
    assert be.gaussians.setup_args == {"training": "args-sentinel"}
    # 平移确实施加在世界 +x
    assert float(be.gaussians._xyz[:, 0].mean()) == pytest.approx(0.3)
    # 两个不变式已重建（否则 is_keyframe 会在 occ_aware_visibility[last_kf] 处 KeyError）
    assert 42 in be.occ_aware_visibility
    assert be.occ_aware_visibility[42].shape[0] == 7
    assert bool(be.occ_aware_visibility[42].all())      # 保守：载入基元默认全可见


# ── 4. 源码接线守卫 ─────────────────────────────────────────────────────────
def test_wiring_before_initialize_map():
    """生效点必须在 init 分支里、且以 enabled 分流（关闭态走原 initialize_map）。"""
    src = (REPO / "utils" / "slam_backend.py").read_text(encoding="utf-8")
    assert "def mapload_enabled(config)" in src
    assert "def mapload_from_ply(self, cur_frame_idx, viewpoint)" in src
    assert "if mapload_enabled(self.config):" in src
    # 分流块里两者都在位，且 else 分支是原 initialize_map
    i = src.find("if mapload_enabled(self.config):")
    assert i > 0
    block = src[i : i + 400]
    assert "self.mapload_from_ply(cur_frame_idx, viewpoint)" in block
    assert "self.initialize_map(cur_frame_idx, viewpoint)" in block
