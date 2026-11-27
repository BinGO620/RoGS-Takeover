"""生命周期状态一致性契约测试（the pre-registered protocol §2 W5）。

判据本体 = the pre-registered protocol.md §3（commit 6279339fb，先于实现）：
- C-J1 契约不变量：全部变异路径后 validate_runtime_state 通过；载入态与
  「extend_from_pcd 刚建 N 个基元」同构；
- 负对照：人为制造基数失配 ⇒ 守卫必须 raise（fail-loud 不可绕过）。

W1 复现（scripts/paper3_bpw/repro_ledger_desync.py，/tmp/repro_ledger_desync.json）
钉死的缺陷 = 载入后 training_setup 被 `optimizer is None` 门跳过 ⇒ 优化器持
add_next_kf 初始图旧张量、load_ply 已换载入新张量 ⇒ 追加时两条谱系分道
（_xyz 10917 vs the internal record 25503，B′ 实测错误逐数吻合）。修复 = mapload 无条件重建
优化器 + kf_anchor 并入 load_ply 本体。本套件守护修复后事实。

GPU 用例（真实张量操作）在无卡环境自动跳过；CPU 用例随时可跑。
用法：/data/conda_envs/monogs-ours/bin/python -m pytest tests/mapload -v
"""
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gaussian_splatting.scene.gaussian_model import GaussianModel, UNTRACKED  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="契约序列需真实 CUDA 张量路径"
)

TRAINING_ARGS = types.SimpleNamespace(
    percent_dense=0.01, position_lr_init=0.0016, feature_lr=0.0025,
    opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001,
    position_lr_final=0.000016, position_lr_delay_mult=0.01,
    position_lr_max_iter=40000, position_lr_max_steps=40000,
)


def _fake_cam():
    return types.SimpleNamespace(
        fx=542.8, fy=542.8, cx=320.0, cy=240.0,
        R=torch.eye(3), T=torch.zeros(3),
    )


def _candidates(n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0, 640, n).astype(np.float32),
        rng.uniform(0, 480, n).astype(np.float32),
        rng.uniform(1.0, 4.0, n).astype(np.float32),
        rng.uniform(0.2, 0.9, (n, 3)).astype(np.float32),
    )


def _cardinality(model):
    names = (
        "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
        "_rotation", "max_radii2D", "xyz_gradient_accum", "denom",
        "unique_kfIDs", "n_obs", "lineage_id",
        "static_prob", "static_obs_count", "unmapped_score",
    )
    return {
        n: int(getattr(model, n).shape[0])
        for n in names
        if isinstance(getattr(model, n), torch.Tensor)
        and getattr(model, n).ndim > 0
    }


def _insert(model, cam, n, kf_id, seed):
    import numpy as np
    px, py, z, color = _candidates(n, seed)
    return model.insert_candidate_gaussians(
        cam, px, py, z, color, kf_id=kf_id,
        lineage_ids=np.full(n, kf_id, dtype=np.int32),
    )


@requires_gpu
def test_load_ply_kf_anchor_first_class_semantics(tmp_path):
    """载入即一等公民：kf_anchor 锚定、int 语义、n_obs=0、lineage=UNTRACKED。"""
    import numpy as np
    src = GaussianModel(0, config={"Dataset": {"point_size": 0.03}})
    src.spatial_lr_scale = 1.0
    src.training_setup(TRAINING_ARGS)
    cam = _fake_cam()
    px, py, z, color = _candidates(200, 1)
    src.insert_candidate_gaussians(
        cam, px, py, z, color, kf_id=7,
        lineage_ids=np.full(200, 7, dtype=np.int32),
    )
    ply = tmp_path / "src.ply"
    src.save_ply(str(ply))

    dst = GaussianModel(0, config={"Dataset": {"point_size": 0.03}})
    dst.load_ply(str(ply), kf_anchor=42)
    assert int(dst.get_xyz.shape[0]) == 200
    assert dst.unique_kfIDs.dtype == torch.int32
    assert bool((dst.unique_kfIDs == 42).all())
    assert int(dst.n_obs.sum()) == 0
    assert bool((dst.lineage_id == UNTRACKED).all())
    # 静态车道随 PLY 恢复，长度与基元一致
    assert dst.static_prob.shape[0] == 200


@requires_gpu
def test_contract_full_chain_green_after_fix(tmp_path):
    """C-J1 主用例：忠实 B′ 链路（修复后）全程 validate 通过。

    链路 = slam.py 空图建优化器 → add_next_kf 初始图 → MapLoad 载入（无条件重建
    优化器）→ KF 插入×2 → prune_lineage reject → prune_nonfinite。
    修复前此链路在首插入即失配（10917 vs 25503，/tmp/repro_ledger_desync.json）。
    """
    model = GaussianModel(0, config={"Dataset": {"point_size": 0.03}})
    model.spatial_lr_scale = 1.0
    model.training_setup(TRAINING_ARGS)                     # slam.py:297 同款
    cam = _fake_cam()

    # add_next_kf 初始图（5822 量级 → 测试用 582）
    import numpy as np
    px, py, z, color = _candidates(582, 0)
    model.insert_candidate_gaussians(
        cam, px, py, z, color, kf_id=0,
        lineage_ids=np.full(582, 0, dtype=np.int32),
    )
    model.validate_runtime_state()                          # 初始图自洽

    # MapLoad 载入（契约路径）：load_ply(kf_anchor) + 无条件 training_setup
    ply = tmp_path / "loaded.ply"
    model.save_ply(str(ply))
    model.load_ply(str(ply), kf_anchor=0)
    model.training_setup(TRAINING_ARGS)                     # 修复本体（无条件）
    model.validate_runtime_state()                          # 载入态自洽
    card = _cardinality(model)
    assert len(set(card.values())) == 1, card               # 15 张量同基数

    # KF 插入 ×2 + reject 删除 + 非有限清理 —— 全程自洽
    n1 = _insert(model, cam, 300, 5, 5)
    assert n1 == 300
    model.validate_runtime_state()
    n2 = _insert(model, cam, 300, 10, 10)
    assert n2 == 300
    model.validate_runtime_state()
    before = int(model.get_xyz.shape[0])
    model.prune_lineage([5])
    assert int(model.get_xyz.shape[0]) == before - 300
    model.validate_runtime_state()
    assert model.prune_nonfinite_points() == 0
    model.validate_runtime_state()
    # 全程无自愈触发（C-J3 的单测级形态）
    assert model.contract_repair_summary() == {
        "static_extend": 0, "static_reset": 0, "lineage_rebuild": 0
    }


@requires_gpu
def test_negative_control_validate_raises_on_desync():
    """负对照：人为制造基数失配 ⇒ 守卫必须 raise（fail-loud 不可绕过）。"""
    model = GaussianModel(0, config={"Dataset": {"point_size": 0.03}})
    model.spatial_lr_scale = 1.0
    model.training_setup(TRAINING_ARGS)
    cam = _fake_cam()
    _insert(model, cam, 100, 0, 2)
    model.validate_runtime_state()                          # 健康态先过
    model.unique_kfIDs = torch.cat(
        (model.unique_kfIDs, torch.zeros(3, dtype=model.unique_kfIDs.dtype))
    )
    with pytest.raises(RuntimeError, match="Gaussian state length mismatch"):
        model.validate_runtime_state()


def test_contract_repair_summary_counters_present():
    """CPU：计数器 getter 存在且默认零（无变异时）。"""
    model = GaussianModel(0, config={"Dataset": {"point_size": 0.03}})
    assert model.contract_repair_summary() == {
        "static_extend": 0, "static_reset": 0, "lineage_rebuild": 0
    }


def _bare_model(n, seed=0):
    """CPU-only GaussianModel with n Gaussians and a small tensor footprint."""
    import numpy as np
    from gaussian_splatting.scene.gaussian_model import build_rotation
    model = GaussianModel(0, config={"Dataset": {"point_size": 0.03}})
    model.spatial_lr_scale = 1.0
    rng = np.random.default_rng(seed)
    xyz = torch.tensor(rng.normal(0, 0.1, (n, 3)), dtype=torch.float32)
    model._xyz = torch.nn.Parameter(xyz.contiguous())
    model._features_dc = torch.nn.Parameter(torch.zeros((n, 1, 3)))
    model._features_rest = torch.nn.Parameter(torch.zeros((n, 15, 3)))
    model._opacity = torch.nn.Parameter(torch.zeros((n, 1)))
    model._scaling = torch.nn.Parameter(torch.zeros((n, 3)))
    model._rotation = torch.nn.Parameter(
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1)
    )
    model.max_radii2D = torch.zeros((n,), dtype=torch.float32)
    model.xyz_gradient_accum = torch.zeros((n, 1), dtype=torch.float32)
    model.denom = torch.zeros((n, 1), dtype=torch.float32)
    model.unique_kfIDs = torch.zeros((n,), dtype=torch.int32)
    model.n_obs = torch.zeros((n,), dtype=torch.int32)
    model.lineage_id = torch.full((n,), UNTRACKED, dtype=torch.int32)
    model.static_prob = torch.full((n, 1), 0.7, dtype=torch.float32)
    model.static_obs_count = torch.zeros((n, 1), dtype=torch.float32)
    model.unmapped_score = torch.zeros((n, 1), dtype=torch.float32)
    return model


def test_repair_counters_increment_positively():
    """正向：自愈**真的触发**时三个计数器必须各自递增（补 G2 重大-9 缺口）。

    2026-11-19：此前测试只断言恒 0，递增分支从未被执行 ⇒ 「counted repairs」这一
    贡献性句缺单测支撑。本测试逐个构造失配并断言计数 +1 且the internal record被重建。
    """
    # (a) static grow ⇒ static_extend +1（保留前缀）
    m = _bare_model(10)
    m.static_prob = m.static_prob[:6]          # 人为制造 grow 失配
    m.static_obs_count = m.static_obs_count[:6]
    m.ensure_static_memory_state()
    assert m.contract_repair_summary()["static_extend"] == 1
    assert m.static_prob.shape[0] == 10

    # (b) static shrink ⇒ static_reset +1（索引对齐不可知，重建）
    m = _bare_model(10)
    m.static_prob = torch.full((14, 1), 0.9)   # the internal record比参数多 ⇒ shrink
    m.static_obs_count = torch.zeros((14, 1))
    m.ensure_static_memory_state()
    assert m.contract_repair_summary()["static_reset"] == 1
    assert m.static_prob.shape[0] == 10

    # (c) prune_points 前 lineage 失配 ⇒ lineage_rebuild +1，且标签全部回到 UNTRACKED
    m = _bare_model(10)
    m.lineage_id = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    m.optimizer = torch.optim.Adam(
        [
            {"params": [m._xyz], "lr": 0.0, "name": "xyz"},
            {"params": [m._features_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [m._features_rest], "lr": 0.0, "name": "f_rest"},
            {"params": [m._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [m._scaling], "lr": 0.0, "name": "scaling"},
            {"params": [m._rotation], "lr": 0.0, "name": "rotation"},
        ]
    )
    mask = torch.zeros((10,), dtype=torch.bool)
    mask[:2] = True                            # 删 2 个 ⇒ 期望 8 个
    m.prune_points(mask)
    assert m.contract_repair_summary()["lineage_rebuild"] == 1
    assert m.lineage_id.shape[0] == 8
    assert bool((m.lineage_id == UNTRACKED).all())
