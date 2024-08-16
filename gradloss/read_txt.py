import numpy as np


def read_txt_data(file_name):
    # 定义空的矩阵
    matrix = []

    # 读取txt文件
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # 按组长度循环读取数据
    group_length = 3
    num_groups = len(lines) // group_length

    for i in range(num_groups):
        start_index = i * group_length
        end_index = (i + 1) * group_length

        # 提取文本行和数据行
        text_line = lines[start_index + 1].strip()
        data_line = lines[start_index + 2].strip()

        # 提取数据
        if data_line.startswith('[') and data_line.endswith(']'):
            data_str = data_line[1:-1]  # 去除方括号
            data_list = data_str.split(',')  # 拆分为单个数据的字符串列表
            data = [float(data_list[j]) for j in range(len(data_list))]  # 提取奇数列数据

            # 将数据添加到矩阵中
            matrix.append(data)

    return matrix



