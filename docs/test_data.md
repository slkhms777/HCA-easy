# 测试数据

下载数据到 ./datasets/
```bash
# 数据托管在 modelscope
wget https://www.modelscope.cn/datasets/slkhms777/HCA-easy_TestData/resolve/master/data/test_data.zip -P ./datasets/
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

- [scripts/example1.sh](../scripts/example1.sh)：多物体示例，数据路径为 `datasets/subject_example1/20231027_112303`
- [scripts/example2.sh](../scripts/example2.sh)：单物体示例，数据路径为 `datasets/subject_example2/card_box3`
