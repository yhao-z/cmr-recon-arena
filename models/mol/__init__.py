"""
Monotone Operator Learning (MOL) — DEQ solver for 2D+T multi-coil MRI.

Reference: https://github.com/migroup-mit/Monotone-Operator-Learning
Paper: "Memory-efficient model-based deep learning with convergence and robustness guarantees"
       A Pramanik, MB Zimmerman, M Jacob, IEEE TCI, 2023.

Only DEQ-related code is migrated (no Unroll/ADMM/gradBlock etc.).

Referenced by pl_modules/mol_module.py (class_path: pl_modules.MOLModule, configs/model/mol.yaml).
"""

from .mol_model import MOL
