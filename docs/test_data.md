# 测试数据

测试数据建议放在 `datasets/` 目录下。`datasets/test_data.zip` 使用 Git LFS 管理。

进入仓库后，可以单独拉取测试数据：

```bash
git lfs pull --include="datasets/test_data.zip"
```

如果仓库已经 clone 下来了，也可以之后再单独拉取测试数据：

```bash
git lfs fetch --include="datasets/test_data.zip"
git lfs checkout datasets/test_data.zip
```

解压test_data.zip到./datasets，文件结果如下：
```text
datasets/
├── calibration/
├── models/
├── subject_example1
└── subject_example2
```


仓库中提供了两个示例脚本：

- [scripts/example1.sh](/mnt/16T/gjx/HO-Cap-Annotation/scripts/example1.sh)：多物体示例，数据路径为 `datasets/subject_example1/20231027_112303`
- [scripts/example2.sh](/mnt/16T/gjx/HO-Cap-Annotation/scripts/example2.sh)：单物体示例，数据路径为 `datasets/subject_example2/card_box3`
