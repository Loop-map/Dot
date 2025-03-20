import numpy as np
import tkinter as tk
from tkinter import Canvas,ttk,Toplevel,messagebox
from PIL import Image, ImageTk
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ImageViewer(tk.Tk):
    def __init__(self, image_path,test_point):
        super().__init__()
        self.title("Image Pixel Coordinates")

        # 加载图片
        self.image = Image.open(image_path)

        # 获取屏幕尺寸
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # 计算缩放比例
        width_ratio = screen_width / self.image.width
        height_ratio = screen_height / self.image.height
        scale_ratio = min(width_ratio, height_ratio)

        # 缩放图片
        new_size = (int(self.image.width * scale_ratio), int(self.image.height * scale_ratio))
        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)  # 使用LANCZOS重采样
        self.photo = ImageTk.PhotoImage(self.image)

        # 获取图像尺寸
        image_width = self.image.width
        image_height = self.image.height

        # 设置窗口的最小尺寸以适应图片尺寸
        self.minsize(image_width, image_height)

        # 创建画布并放置图片，画布大小设置为图像大小
        self.canvas = Canvas(self, width=image_width, height=image_height)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.canvas.pack()

        # 用于存储坐标的字典和额外点击的像素坐标列表
        self.coordinates = {}
        self.P_A = []
        self.P_A_copy = []
        self.points_entered = 0
        self.insert_num = 0
        # 控制是否允许弹出新对话框的标志
        self.allow_new_dialog = True
        # 存储当前的对话框，以便可以销毁它
        self.current_dialog = None
        # 存储列名
        self.curve_names = []
        # 创建一个列表来存储画布上的点对象
        self.points_on_canvas = []
        # 插值点间隔和白噪声方差
        self.default_a = 5  # 默认插值点间隔
        self.a = self.default_a  # 当前使用的插值点间隔
        self.scale = 0  # 白噪声方差
        self.scale_adjusted = False  # 标志，表示是否调整了白噪声方差
        # 用于存储每次生成的self.P_A的列表
        self.P_A_history = []
        self.current_curve_index = 0  # 当前正在收集的曲线索引
        self.curves_to_collect = 0  # 用户需要识别的曲线个数
        self.end_button = None  # "end" 按钮
        self.test_point_dict = test_point

        # 创建 "end" 按钮
        self.end_button = ttk.Button(self, text="End", command=self.end_curve_collection)
        self.end_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if not self.allow_new_dialog and self.current_dialog:
            messagebox.showerror("操作错误", "请先关闭当前的输入框或完成输入。")
            return

        # 检查是否已输入三个点
        if self.points_entered >= 3:
            if self.current_curve_index == 0 and len(self.P_A_copy) == 0:
                # 弹出对话框让用户输入识别曲线的个数
                self.input_curve_count()
            # 存储当前点击的点
            self.P_A_copy.append((event.x, event.y))
            self.draw_point(event.x, event.y, 'red')
            if len(self.P_A_copy) >= 2:
                self.interpolate_points(0, 1, self.a, f'interpolated_points_{self.insert_num}')
                self.adjust_interpolation()
                self.wait_window(self.current_dialog)  # 等待对话框关闭

        else:
            self.edit_coordinate(event.x, event.y)


    # def end_curve_collection(self):
    #     # 将当前收集的点添加到历史记录中
    #     self.P_A_history.append(self.P_A[:])
    #     self.P_A.clear()  # 清空当前曲线的点
    #     self.P_A_copy.clear()
    #     self.current_curve_index += 1
    #
    #     if self.current_curve_index < self.curves_to_collect:
    #         curve_name_dialog = Toplevel(self)
    #         curve_name_dialog.title("Input Curve Name")
    #
    #         ttk.Label(curve_name_dialog, text="请输入曲线的名称：").pack(pady=5)
    #         name_entry = ttk.Entry(curve_name_dialog, width=10)
    #         name_entry.pack(pady=5)
    #
    #         def on_submit_curve_name():
    #             curve_name = name_entry.get()  # 获取曲线名称
    #             self.curve_names.append(curve_name)  # 添加到列表
    #             name_entry.destroy()
    #             curve_name_dialog.destroy()  # 关闭当前对话框
    #             messagebox.showinfo("信息", "请继续收集下一条曲线")
    #
    #         name_entry.bind("<Return>", lambda event: on_submit_curve_name())
    #         name_entry.focus()  # 聚焦输入框
    #     else:
    #         curve_name_dialog = Toplevel(self)
    #         curve_name_dialog.title("Input Curve Name")
    #
    #         ttk.Label(curve_name_dialog, text="请输入曲线的名称：").pack(pady=5)
    #         name_entry = ttk.Entry(curve_name_dialog, width=10)
    #         name_entry.pack(pady=5)
    #
    #         def on_submit_curve_name():
    #             curve_name = name_entry.get()  # 获取曲线名称
    #             self.curve_names.append(curve_name)  # 添加到列表
    #             name_entry.destroy()
    #             curve_name_dialog.destroy()  # 关闭当前对话框
    #             messagebox.showinfo("完成", "所有曲线收集完成")
    #
    #         name_entry.bind("<Return>", lambda event: on_submit_curve_name())
    #         name_entry.focus()  # 聚焦输入框

    def end_curve_collection(self):
        # 将当前收集的点添加到历史记录中
        self.P_A_history.append(self.P_A[:])
        self.P_A.clear()  # 清空当前曲线的点
        self.P_A_copy.clear()
        self.current_curve_index += 1

        if self.current_curve_index < self.curves_to_collect:
            curve_name_dialog = Toplevel(self)
            curve_name_dialog.title("Select Curve Name")

            # 创建下拉菜单
            curve_names = list(self.test_point_dict.values())
            curve_name_label = ttk.Label(curve_name_dialog, text="请选择曲线的名称：")
            curve_name_label.pack(pady=5)
            curve_name_combobox = ttk.Combobox(curve_name_dialog, values=curve_names, state="readonly")
            curve_name_combobox.pack(pady=5)
            curve_name_combobox.current(0)  # 默认选择第一个选项

            # 创建确认按钮
            confirm_button = ttk.Button(curve_name_dialog, text="确认",
                                        command=lambda: on_submit_curve_name(curve_name_combobox.get()))
            confirm_button.pack(pady=5)

            def on_submit_curve_name(selected_name):
                self.curve_names.append(selected_name)  # 添加到列表
                curve_name_dialog.destroy()  # 关闭当前对话框
                messagebox.showinfo("信息", "请继续收集下一条曲线")

        else:
            curve_name_dialog = Toplevel(self)
            curve_name_dialog.title("Select Curve Name")

            # 创建下拉菜单
            curve_names = list(self.test_point_dict.values())
            curve_name_label = ttk.Label(curve_name_dialog, text="请选择曲线的名称：")
            curve_name_label.pack(pady=5)
            curve_name_combobox = ttk.Combobox(curve_name_dialog, values=curve_names, state="readonly")
            curve_name_combobox.pack(pady=5)
            curve_name_combobox.current(0)  # 默认选择第一个选项

            # 创建确认按钮
            confirm_button = ttk.Button(curve_name_dialog, text="确认",
                                        command=lambda: on_submit_curve_name(curve_name_combobox.get()))
            confirm_button.pack(pady=5)

            def on_submit_curve_name(selected_name):
                self.curve_names.append(selected_name)  # 添加到列表
                curve_name_dialog.destroy()  # 关闭当前对话框
                messagebox.showinfo("完成", "所有曲线收集完成")



    def input_curve_count(self):
        if self.current_dialog:
            return  # 如果对话框已经存在，则不做任何操作
        self.current_dialog = Toplevel(self)
        self.current_dialog.title("Input Curve Count")

        ttk.Label(self.current_dialog, text="请输入识别曲线的个数：").pack(pady=5)
        curve_count_entry = ttk.Entry(self.current_dialog, width=10)
        curve_count_entry.pack(pady=5)

        def on_confirm_curve_count():
            try:
                self.curves_to_collect = int(curve_count_entry.get())
                if self.curves_to_collect <= 0:
                    raise ValueError("曲线个数必须大于0")
                self.current_dialog.destroy()
                self.current_dialog = None
                self.end_button.pack(side=tk.RIGHT, padx=10, pady=10)  # 显示 "end" 按钮
            except ValueError:
                messagebox.showerror("输入错误", "请输入有效的数字！")

        ttk.Button(self.current_dialog, text="确认", command=on_confirm_curve_count).pack(pady=10)
        self.current_dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)


    def add_noise_to_points(self, scale, tags=None):
        if len(self.P_A_copy) < 2:
            print("至少需要两个点以添加噪声。")
            return

        # 获取最近添加的两个点（最后一个点和它之前的点）
        self.interpolate_points(len(self.P_A_copy) - 2, len(self.P_A_copy) - 1, self.a, tags)
        self.canvas.delete(tags)

        noise_points = [p for p in self.P_A_copy[1:len(self.P_A_copy) - 1] if isinstance(p, tuple) and len(p) == 2]
        _, y_values = zip(*noise_points)

        # 生成白噪声，均值使用 Y 坐标的均值，方差使用 Y 坐标的方差
        noise = np.random.normal(loc=0, scale=scale, size=len(noise_points))

        # 将白噪声加到插值点的 Y 坐标上，X 坐标保持不变
        for i, (x, _) in enumerate(noise_points):
            noise_y = y_values[i] + noise[i]  # 确保噪声值在合理的范围内
            self.P_A_copy[i + 1] = (x, noise_y)
            self.draw_point(x, noise_y, 'green', tags)  # 使用绿色表示加了噪声的点
        # print(self.P_A_copy)

        if len(noise_points) > 1:
            points_to_connect = [point for point in self.P_A_copy[1:len(self.P_A_copy) - 1] if
                                 isinstance(point, tuple) and len(point) == 2]
            for i in range(1, len(points_to_connect)):
                self.canvas.create_line(
                    points_to_connect[i - 1][0], points_to_connect[i - 1][1],
                    points_to_connect[i][0], points_to_connect[i][1],
                    fill='green', tags=tags
                )

    def interpolate_points(self, start_index, end_index, a=10, tags=None):
        if start_index < 0 or end_index >= len(self.P_A_copy) or end_index <= start_index:
            return  # 检查索引是否有效

        p1 = np.array(self.P_A_copy[start_index])
        p2 = np.array(self.P_A_copy[end_index])
        dist = np.linalg.norm(p2 - p1)
        num_interpolate = max(0, int(dist / a))  # 计算需要插值的点的数量

        # 插入点
        step_x = (p2[0] - p1[0]) / (num_interpolate + 1)
        step_y = (p2[1] - p1[1]) / (num_interpolate + 1)
        interp_points = [p1 + (step_x * (j + 1), step_y * (j + 1)) for j in range(num_interpolate)]
        interp_points_sorted = sorted(interp_points, key=lambda point: point[0], reverse=True)

        # 将新插入的点添加到P_A中，并绘制到画布上
        for interp_point in interp_points_sorted:
            self.P_A_copy.insert(start_index + 1, tuple(interp_point))
            # print(self.P_A_copy)
            self.draw_point(*interp_point, 'blue', tags)

    def draw_point(self, x, y, color, tags=None):
        # 绘制一个点到画布上
        radius = 2  # 点的半径
        point = self.canvas.create_oval(x - radius, y - radius,
                                        x + radius, y + radius,
                                        fill=color, outline="")
        if tags:
            # 给点打标签，如果提供了标签
            self.canvas.addtag_withtag(tags, point)
        # 存储点对象以便后续需要时可以更新或删除
        self.points_on_canvas.append(point)

    def adjust_interpolation(self):
        if self.current_dialog:
            return  # 如果对话框已经存在，则不做任何操作

        self.allow_new_dialog = False  # 禁用新的对话框
        self.current_dialog = Toplevel(self)
        self.current_dialog.title("Adjust Interpolation and Noise")

        # 使用 grid 布局
        row = 0

        # 间隔调整行
        ttk.Label(self.current_dialog, text="间隔：").grid(row=row, column=0, padx=2, pady=2, sticky='w')
        self.a_label = ttk.Label(self.current_dialog, width=3, text=str(self.a))
        self.a_label.grid(row=row, column=1, padx=2, pady=2, sticky='e')
        ttk.Button(self.current_dialog, text="↑", width=2, command=lambda: self.adjust_parameter('a', 1)).grid(row=row,
                                                                                                               column=2,
                                                                                                               padx=1,
                                                                                                               pady=2)
        ttk.Button(self.current_dialog, text="↓", width=2, command=lambda: self.adjust_parameter('a', -1)).grid(row=row,
                                                                                                                column=3,
                                                                                                                padx=1,
                                                                                                                pady=2)

        row += 1  # 移到下一行

        # 白噪声调整行
        ttk.Label(self.current_dialog, text="白噪声：").grid(row=row, column=0, padx=2, pady=2, sticky='w')
        self.scale_label = ttk.Label(self.current_dialog, width=3, text=str(self.scale))  # 调整宽度以匹配间隔标签
        self.scale_label.grid(row=row, column=1, padx=2, pady=2, sticky='e')
        ttk.Button(self.current_dialog, text="↑", width=2, command=lambda: self.adjust_parameter('scale', 0.5)).grid(
            row=row, column=2, padx=1, pady=2)
        ttk.Button(self.current_dialog, text="↓", width=2, command=lambda: self.adjust_parameter('scale', -0.5)).grid(
            row=row, column=3, padx=1, pady=2)

        # 确定按钮
        ttk.Button(self.current_dialog, text="确定", command=self.on_confirm_adjust).grid(row=row + 1, column=0,
                                                                                        columnspan=4, pady=5,
                                                                                        sticky='ew')

        self.current_dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)

        # 使对话框可拖动
        self.current_dialog.transient(self)  # 使对话框模态
        self.current_dialog.grab_set()  # 使对话框成为前台窗口
        self.current_dialog.wait_window(self.current_dialog)  # 阻塞直到对话框被关闭

        # 可以添加一些列配置来确保所有列都有相同的宽度
        # for col in range(4):
        #     self.current_dialog.grid_columnconfigure(col, weight=2)

    def adjust_parameter(self, param, value):
        if param == 'a':
            # self.a = max(1, int(self.a_entry.get()) + value)
            # self.a_entry.delete(0, tk.END)
            # self.a_entry.insert(0, str(self.a))
            self.a = max(1, int(self.a) + value)
            self.a_label.config(text=str(self.a))  # 更新 Label 显示的值
        elif param == 'scale':
            # self.scale = float(self.scale_entry.get()) + value
            # self.scale_entry.delete(0, tk.END)
            # self.scale_entry.insert(0, str(self.scale))
            self.scale = float(self.scale) + value
            self.scale_label.config(text=str(self.scale))
            self.scale_adjusted = True  # 用户已经调整了白噪声方差

        self.redraw_points_and_lines()

    def redraw_points_and_lines(self):
        # 清除之前的点和线
        self.canvas.delete(f'interpolated_points_{self.insert_num}')
        self.P_A_copy = [self.P_A_copy[0], self.P_A_copy[-1]]
        # 重新绘制点和线
        if self.scale_adjusted:
            if self.scale < 0:
                messagebox.showerror("错误", "方差必须大于0")
                return  # 退出方法，不执行后续的添加噪声操作
            else:
                self.add_noise_to_points(self.scale, f'interpolated_points_{self.insert_num}')
        else:
            self.interpolate_points(0, len(self.P_A_copy) - 1, self.a, f'interpolated_points_{self.insert_num}')

    def on_confirm_adjust(self):
        # 使用最后一次调整的参数重新绘制
        self.P_A.extend(self.P_A_copy)  # 将临时列表的数据移动到主列表
        self.P_A_copy.clear()  # 清空临时列表
        self.P_A_copy.append(self.P_A[-1])  # 保留最后一个点以便下次操作
        self.insert_num = self.insert_num + 1
        self.a = self.default_a
        self.scale = 0
        self.current_dialog.destroy()
        self.current_dialog = None
        self.allow_new_dialog = True
        self.scale_adjusted = False

    def edit_coordinate(self, x, y):
        if self.current_dialog:
            # 如果已经存在对话框，不创建新的
            return

        self.current_dialog = Toplevel(self)
        self.current_dialog.title(f"Enter Coordinates for Point {self.points_entered + 1}")
        ttk.Label(self.current_dialog, text=f"Initial Click at ({x}, {y}):").pack(pady=5)

        ttk.Label(self.current_dialog, text="User X-Coor:").pack(pady=5)
        x_entry = ttk.Entry(self.current_dialog, width=10)
        x_entry.pack(pady=5)
        ttk.Label(self.current_dialog, text="User Y-Coor:").pack(pady=5)
        y_entry = ttk.Entry(self.current_dialog, width=10)
        y_entry.pack(pady=5)

        def on_confirm():
            user_x = x_entry.get().strip()
            user_y = y_entry.get().strip()
            if user_x and user_y:
                try:
                    coord_x = float(user_x)
                    coord_y = float(user_y)
                    self.coordinates[(coord_x, coord_y)] = (x, y)
                    self.points_entered += 1
                    self.allow_new_dialog = True
                    self.current_dialog.destroy()
                    self.current_dialog = None
                except ValueError:
                    messagebox.showerror("输入错误", "请输入有效的数字！")
            else:
                messagebox.showerror("输入错误", "请输入x，y值！")

        ttk.Button(self.current_dialog, text="Confirm", command=on_confirm).pack(pady=10)
        self.current_dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def on_cancel(self):
        # 当用户尝试关闭对话框时调用
        # 不改变 self.allow_new_dialog 的状态，允许用户在关闭对话框后继续操作
        self.current_dialog.destroy()
        self.current_dialog = None

    def get_coordinates(self):
        # 返回当前存储的坐标
        return self.coordinates.copy()  # 返回一个副本

    def get_PA(self):
        # 返回额外点击的像素坐标列表 P_A
        return self.P_A.copy()  # 返回一个副本

    def get_PA_his_list(self):
        # 返回识别曲线个数个像素坐标列表所组成的列表
        return self.P_A_history.copy()

    def affine_transform(self,points_src, points_dst):
        assert len(points_src) >= 3, "At least 3 points are required."
        assert points_src.shape == points_dst.shape and points_src.shape[
            1] == 2, "Points must be 2D and of the same shape."

        num_points = len(points_src)
        A = np.zeros((2 * num_points, 6))
        b = np.zeros(2 * num_points)

        for i in range(num_points):
            x_src, y_src = points_src[i]
            x_dst, y_dst = points_dst[i]
            A[2 * i] = [x_src, y_src, 1, 0, 0, 0]  # x 的系数
            A[2 * i + 1] = [0, 0, 0, x_src, y_src, 1]  # y 的系数
            b[2 * i] = x_dst
            b[2 * i + 1] = y_dst

        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        a, b, tx, c, d, ty = params

        transform_matrix = np.array([[a, b, tx],
                                     [c, d, ty],
                                     [0, 0, 1]])

        return transform_matrix

    def transform_point(self,points_src, points_dst, new_point):
        """
        使用仿射变换矩阵计算新点的映射坐标。

        参数:
        - points_src: 原始点集，用于计算变换矩阵。
        - points_dst: 目标点集，与points_src对应，用于计算变换矩阵。
        - new_point: 原始坐标系中的点，形状为 (2,) 或 (2, 1)。

        返回:
        - transformed_point: 变换后的点坐标，形状为 (1, 2)。
        """
        # 计算仿射变换矩阵
        transform_matrix = self.affine_transform(points_src, points_dst)

        # 将点转换为齐次坐标形式，确保它是 (1, 3)
        point_homogeneous = np.hstack([new_point.reshape(1, -1), np.array([[1]])])

        # 应用仿射变换矩阵
        transformed_point_homogeneous = point_homogeneous @ transform_matrix.T  # 注意这里可能需要转置

        # 将齐次坐标转换回二维坐标，并转换为 (1, 2) 形状
        transformed_point = transformed_point_homogeneous[:, :2].reshape(1, 2)

        return transformed_point

    def interpolate_df_to_target_length(self,df,target_length):
        """
        对一个DataFrame进行插值，使得y列的长度等于target_length。
        假设x列是排序的，但不等间隔。

        参数:
        - df: 一个Pandas DataFrame，包含x和y列。
        - target_length: 目标长度，即插值后y列的长度。

        返回:
        - interpolated_df: 插值后的DataFrame。
        """
        # 获取第一列列名
        first_column_name = df.columns[0]
        # 获取第二列列名
        second_column_name = df.columns[1]

        # 确保x列是排序的
        df = df.sort_values(first_column_name)

        # 计算x的最小值和最大值
        x_min, x_max = df[first_column_name].min(), df[first_column_name].max()

        # 根据目标长度和x的范围计算步长
        step = (x_max - x_min) / (target_length - 1)

        # 创建一个等间隔的xnew序列
        xnew = np.arange(x_min, x_max + step, step)

        # 使用interp1d进行插值
        f = interp1d(df[first_column_name], df[second_column_name], kind='linear', fill_value="extrapolate")
        ynew = f(xnew)

        # 创建一个新的DataFrame
        interpolated_df = pd.DataFrame({first_column_name: xnew, second_column_name: ynew})
        return interpolated_df


import pandas as pd
# 确保代码在主程序中运行
if __name__ == "__main__":
    image_path = '../data/气流激振故障.png'  # 替换为你的图片路径
    # file_path = 'ident_data/real_data.csv'
    test_point_dic = {
          "sx01032": "主轴密封清洁水源流量低",
          "sx01033": "主轴密封水温(报警)高",
          "sx01034": "主轴密封水温(报警)过高",
          "sx01035": "主轴密封压差高",
          "sx01038": "主轴密封技术供水流量低",
          "sx01039": "主轴密封水OK",
          "sx01040": "主轴密封清洁水源流量<12",
          "sx01041": "主轴密封水流量<9",
          "sx01042": "主轴密封技术供水流量<10.8",
          "sx01043": "主轴密封供水压力>0.95",
          "sx01044": "主轴密封供水压力<0.4",
          "sx01045": "主轴密封清洁水源流量",
          "sx01046": "主轴密封技术供水流量",
          "sx01047": "主轴密封水流量",
          "sx01048": "主轴密封供水压力",
          "sx01061": "主轴密封水水温2",
          "sx01062": "主轴密封水进水口压力",
          "sx01064": "主轴密封水水温1",
          "sx01065": "主轴密封水出水口压力",
          "sx01058": "主轴密封磨损报警",
          "sx00030": "顶盖液位"
        }
    app = ImageViewer(image_path,test_point_dic)
    app.mainloop()
    data_dict = app.get_coordinates()
    column_names = app.curve_names
    print(column_names,'column_names')
    real_P = list(data_dict.keys())
    # 提取值到列表 B
    picture_P = list(data_dict.values())
    total_real_data_list = []
    picture_P_array = np.array(picture_P)
    real_P_array = np.array(real_P)
    dfs = []
    dfs_dic = {}
    total_picture_list = app.get_PA_his_list()
    for total_picture_data in total_picture_list:
        real_data_list = []
        for P_A in total_picture_data:
            point_old = np.array(P_A)
            real_point = app.transform_point(picture_P_array, real_P_array, point_old)
            real_data_list.append(real_point)
        total_real_data_list.append(real_data_list)
    for i, sublist in enumerate(total_real_data_list):
        # 初始化一个空字典来存储该子列表的坐标数据
        sub_data = []
        # 遍历子列表中的每个坐标数组
        for coord in sublist:
            x, y = coord[0]
            sub_data.append({f'time_{i}': x, column_names[i]: y})
            # 使用sub_data列表创建DataFrame，并添加到dfs列表中
        df = pd.DataFrame(sub_data)
        dfs.append(df)
    if len(dfs) > 1:
        max_length = max(df.shape[0] for df in dfs)
    else:
        max_length = dfs[0].shape[0]
    # 打印每个DataFrame查看结果
    for i, df in enumerate(dfs):
        interpolated_df = app.interpolate_df_to_target_length(df, max_length)
        dfs_dic[f'DataFrame {i + 1}'] = interpolated_df
    result_df = pd.concat([dfs_dic[key] for key in sorted(dfs_dic.keys())], axis=1)
    print(result_df)
    result_df.to_csv('../data/create_data.csv', index=False)
   y_cols = [col for col in result_df.columns if 'time' not in col]
    # 设置图像大小
    plt.figure(figsize=(10, 6))
    # 遍历 y 列，绘制每个 y 列对 time_0 的图形
    for col in y_cols:
        plt.plot(result_df['time_0'], result_df[col], label=col)
    # 设置 x 轴和 y 轴标签
    plt.xlabel('time_0')  # x 轴标签
    plt.ylabel('Values')  # y 轴标签
    # 设置图像标题
    plt.title('Plot of Columns against time_0')
    # 显示图例
    plt.legend()
    # 显示网格
    plt.grid(True)
    # 显示图像
    plt.show()







