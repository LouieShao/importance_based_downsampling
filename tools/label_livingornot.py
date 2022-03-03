import json
with open("/home/shaoshihao/ipython_file/simpleAICV-pytorch-ImageNet-COCO-training-master/tools/label_to_content_categories.json",'r') as load_f:
    load_dict = json.load(load_f)
living_keys = list(load_dict['living'].keys())
nonliving_keys = list(load_dict['non_living'].keys())
