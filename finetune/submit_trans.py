import numpy as np
import json
import argparse

def read_input_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default='IMCS-DAC_test_submit.json', type=str)

    config = parser.parse_args()
    output_path = config.output_file

    tags = [
        'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose',
        'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice',
        'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions',
        'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom',
        'Inform-Symptom', 'Other'
    ]

    init_data = read_input_file("/home/luguangyue/lgy/医疗对话意图检测/bert+crf/data/IMCS-DAC_test.json")
    data = np.load('./output_data/predict_data.npz')
    test_label_ids = data['test_prediction']
    no_id = 0 # 从前往后依次去label
    for k in init_data.keys():
        for i in range(len(init_data[k])):
            sentence = init_data[k][i]
            sentence_id = sentence["sentence_id"]
            sentence["dialogue_act"] = tags[test_label_ids[no_id]]
            no_id += 1
    with open(f"/home/luguangyue/lgy/医疗对话意图检测/finetune/output_data/{output_path}", 'w', encoding='utf-8') as json_file:
        json.dump(init_data, json_file, ensure_ascii=False, indent=4)
