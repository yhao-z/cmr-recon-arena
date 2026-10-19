# Configs

配置拆分为两部分，通过多个 `--config` 传给 main.py（按键深层合并，**后面的覆盖前面的**）：

```bash
python main.py fit --config configs/data/<dataset>.yaml --config configs/<model>.yaml
```

- `data/`：数据配置，只含 `data:` 段（路径、mask、acceleration、test_split 等）→ 换数据集只换这个文件
- `<model>.yaml`：模型配置，含 `seed_everything` / `trainer` / `model`，无 `data:` 段（续训时自行加 `ckpt_path`）
  （顶部注释 `# data: ...` 标明拆分前配套的数据配置）

## 数据配置清单

| 文件 | 用途 |
|------|------|
| `data/ocmr_lowacc.yaml` | OCMR 训练，GROMaskFunc accel 6-10 |
| `data/ocmr_highacc.yaml` | OCMR 训练，GROMaskFunc accel 18-22 |
| `data/cmrxrecon23.yaml` | CMRxRecon23 训练，accel 4/8/10 |
| `data/ocmr_test_lowacc.yaml` | OCMR 测试，test_gro_06/08/10 |
| `data/ocmr_test_highacc.yaml` | OCMR 测试，test_gro_18/20/22 |
| `data/cmrxrecon23_test.yaml` | CMRxRecon23 测试，test_pse_04/08/10 |

单个 data 字段可在命令行临时覆盖：`--data.init_args.num_workers 8`、
`--data.init_args.test_split '["test_gro_08"]'`。

子目录 `cmrxrecon23cine/`、`test/` 下的文件已全部去掉 `data:` 段，配套数据配置见各自顶部注释。
