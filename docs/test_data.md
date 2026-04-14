# 测试数据

测试数据建议放在 `datasets/` 目录下。`datasets/test_data.zip` 使用 Git LFS 管理。

进入仓库后，单独拉取测试数据：

```bash
git lfs pull --include="datasets/test_data.zip"
```

或者使用：

```bash
git lfs fetch --include="datasets/test_data.zip"
git lfs checkout datasets/test_data.zip
```

解压test_data.zip到./datasets
```bash
mkdir temp_dir
unzip datasets/test_data.zip -d temp_dir
mv temp_dir/test_data/* datasets/
rm -r temp_dir
```

文件结果如下：
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
