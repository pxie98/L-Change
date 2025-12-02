
import scipy.io as sio

mat_file_path = 'func_video/approx1/data/vorticity.mat'
mat_contents = sio.loadmat(mat_file_path)
data=mat_contents["vorticity"]
processed_data = data[0:10,126:137,25:81]


import matplotlib.pyplot as plt
for i in range(10):
    # 绘制二维图
    plt.imshow(processed_data[i]/40, cmap='viridis', aspect='auto', vmin=-1., vmax=1.)  # cmap 是颜色映射，aspect='auto' 自动调整宽高比
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('Value')  # 设置颜色条的标签
    # 添加标题和坐标轴标签
    plt.title("2D Matrix Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # 显示图形
    plt.show()
    fig_name = 'func_video/approx1/data/fitpngraw_{}.png'.format(i)
    plt.savefig(fig_name)
    plt.close()




output_file_path = 'func_video/approx1/data/rawfig/vorticity_small.mat'
sio.savemat(output_file_path, {'vorticity_small': processed_data})


