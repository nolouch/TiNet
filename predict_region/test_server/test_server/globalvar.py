class gloVar:
    data = {
        'time': 0,
        'predict_step': 0,
        'history_r2_score_total': 0,
        'table_num': 0,
        'table_info': [],
        # 'history_r2_score': [],
        # 'maxvalue': [],
        # 'minvalue': [],
        # 'key_range':[],
        # 'predict': []
    }
def set_demo_value(value):
    gloVar.data = value
def get_demo_value():
    return gloVar.data
