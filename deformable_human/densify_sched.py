# -*- coding: utf-8 -*-
"""方案 D 稠密化调度器的 deformable_human 侧入口。

实现本体在 ../som_human/densify_sched.py（单一事实来源，避免双份漂移）。
⚠️ 不能用「sys.path 加 som_human + import densify_sched」：本文件与目标模块
同名（densify_sched），import 时会命中 sys.modules 里的自己 → 循环导入。
这里用 importlib 按文件路径加载到独立模块名。
"""
import importlib.util as _ilu
import os as _os

_PATH = _os.path.normpath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "som_human",
    "densify_sched.py"))
if not _os.path.exists(_PATH):
    raise ImportError("som_human/densify_sched.py 不存在: {}".format(_PATH))

_spec = _ilu.spec_from_file_location("som_densify_sched", _PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DensifyScheduler = _mod.DensifyScheduler
sched_densify_and_prune = _mod.sched_densify_and_prune
sched_prune_only = _mod.sched_prune_only
grad_scores = _mod.grad_scores

__all__ = ["DensifyScheduler", "sched_densify_and_prune", "sched_prune_only",
           "grad_scores"]
