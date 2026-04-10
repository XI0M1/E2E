import json

# 解析knob配置文件
def get_knobs(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
