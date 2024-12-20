## 模型训练与预测
# 1. 训练模型
# 参数：{train_file}: 训练数据集路径，{dev_file}: 验证数据集路径，{pretrained_model_dir}: 预训练语言模型路径，{output_model_dir}: 模型保存路径
python trainer.py --train_file ./data/IMCS_train.json --dev_file ./data/IMCS_dev.json --pretrained_model_dir ./plms/chinese_bert_wwm --output_model_dir ./save_model/chinese_bert_wwm --cuda_id cuda:0 --batch_size 1 --num_epochs 10 --patience 3
# 2. 预测
# 参数：{test_input_file}: 测试数据集路径，{test_output_file}: 预测结果输出路径，{model_dir}: 加载已训练模型的路径，{pretrained_model_dir}: 预训练语言模型的路径
python predict.py --test_input_file ./data/IMCS_test.json --test_output_file IMCS-IR_test.json --model_dir ./save_model/chinese_bert_wwm --pretrained_model_dir ./plms/chinese_bert_wwm --cuda_id cuda:0
# 3. 提交文件格式转换
python submit_trans.py --data_file IMCS-DAC_test.json --output_file IMCS-DAC_test_submit.json