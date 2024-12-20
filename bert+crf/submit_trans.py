import json
import argparse

def read_input_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", default='IMCS-DAC_test.json', type=str)
    parser.add_argument("--output_file", default='IMCS-DAC_test_submit.json', type=str)

    config = parser.parse_args()
    pred_path = config.data_file
    output_path = config.output_file

    init_data = read_input_file("/home/luguangyue/lgy/医疗对话意图检测/bert+crf/data/IMCS-DAC_test.json")
    predict_data = read_input_file(f"/home/luguangyue/lgy/医疗对话意图检测/bert+crf/output_data/{pred_path}")
    for k in init_data.keys():
        for i in range(len(init_data[k])):
            sentence = init_data[k][i]
            sentence_id = sentence["sentence_id"]
            sentence["dialogue_act"] = predict_data[k][sentence_id]
    with open(f"/home/luguangyue/lgy/医疗对话意图检测/bert+crf/output_data/{output_path}", 'w', encoding='utf-8') as json_file:
        json.dump(init_data, json_file, ensure_ascii=False, indent=4)
