import pandas as pd
import json
import PySimpleGUI as sg #графический интерфейс
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

from statsmodels.tsa.seasonal import seasonal_decompose

def get_data_trend(arr_data):
    arr_data_trend = []
    for arr in arr_data:
        arr_data_trend += [seasonal_decompose(arr, period=12).trend[6:-6]]
    return np.array(arr_data_trend)

def get_data_without_trend(arr_data):
    arr_data_without_trend = []
    for arr in arr_data:
        arr_data_without_trend += [arr[6:-6] - seasonal_decompose(arr, period=12).trend[6:-6]]
    return np.array(arr_data_without_trend)

def normalize(df, min_border, max_border, norm_lst, is_normal_rasp):
    for i, feat in enumerate(norm_lst):
        vals = df[feat].values
        if is_normal_rasp[i]:
            df[feat] = 2*(vals - min_border[feat]) / (max_border[feat] - min_border[feat]) - 1
        else:
            df[feat] = vals / max_border[feat]

sg.theme('SystemDefaultForReal')
layout = [[ sg.Input(), sg.FileBrowse('Выбрать файл')],[sg.Submit(), sg.Cancel()]]
layout1 = [[sg.Text("Неверный формат файла, перезагрузите программу", font='Courier 14', text_color='black')], [sg.Button("OK")]]


window = sg.Window('Импорт данных', layout)

while True:
    event, values = window.read()
    print('Путь к файлу', values['Выбрать файл'])
     
    if event is None or event == 'Cancel':
        break
    
    if event == 'Submit':
        break
    if event in (None, 'Exit', 'Cancel'):
        break
if (Path(values['Выбрать файл']).suffix == '.csv') == True:
    

    print('Идет обработка данных')
    df = pd.read_csv(values['Выбрать файл'], delimiter=';')

    assert df.loc[df.Data == '[]'].shape[0] == 0, 'Имеются пустые списки данных с сенсоров'
    assert df.loc[df.Data_2 == '[]'].shape[0] == 0,'Имеются пустые списки данных с сенсоров'

    str_data = df.loc[:, ['Data', 'Data_2']].values

    lst_data_1 = []
    lst_data_2 = []
    for it in str_data:
        lst_1 = list(it[0][1:-1].replace(' ', '').split(','))
        lst_2 = list(it[1][1:-1].replace(' ', '').split(','))

        lst_data_1.append(lst_1)
        lst_data_2.append(lst_2)
    

    mask_1 = np.array(list(map(len, lst_data_1))) == 240
    mask_2 = np.array(list(map(len, lst_data_2))) == 240    
    mask = mask_1 * mask_2
    
    assert sum(mask) == df.shape[0], 'Некоторые данные с сенсоров имеют неверный размер'

    lst_data_1 = np.array(lst_data_1, np.float64)
    lst_data_2 = np.array(lst_data_2, np.float64)

    df['lst_data1'] = [*lst_data_1]
    df['lst_data2'] = [*lst_data_2]
    
    df = df.drop(columns=['Data', 'Data_2'])
    df = df.reset_index(drop=True)

    print('Производим выделение параметров')

    new_data = get_data_trend(lst_data_1)
    new_data_wt = get_data_without_trend(lst_data_1)

    new_data_2 = get_data_trend(lst_data_2)
    new_data_wt_2 = get_data_without_trend(lst_data_2)
    
    n = len(new_data[0])

    X = pd.DataFrame({  'Test_index':df.Test_index.values/6, 'Presentation':df.Presentation.values/4, 'Question':df.Question.values/13, 
                        'd1_std':new_data.std(axis=1),
                        'd1_mean':new_data.mean(axis=1),
                        'd1_median':np.median(new_data,axis=1),
                        'd1_minmax':new_data.max(axis=1)-new_data.min(axis=1),
                        'd1_std_1':new_data[:,:n//2].std(axis=1),
                        'd1_mean_1':new_data[:,:n//2].mean(axis=1),
                        'd1_median_1':np.median(new_data[:,:n//2],axis=1),
                        'd1_max_1':new_data[:,:n//2].max(axis=1),
                        'd1_std_2':new_data[:,n//2:].std(axis=1),
                        'd1_mean_2':new_data[:,n//2:].mean(axis=1),
                        'd1_median_2':np.median(new_data[:,n//2:],axis=1),
                        'd1_min_2':new_data[:,n//2:].min(axis=1),
                        'd1_std_wt':new_data_wt.std(axis=1),
                        'd1_mean_wt':new_data_wt.mean(axis=1),
                        'd1_median_wt':np.median(new_data_wt,axis=1),
                        'd1_min_wt':new_data_wt.max(axis=1)-new_data_wt.min(axis=1),
                        'd1_std_wt_1':new_data_wt[:,:n//2].std(axis=1),
                        'd1_mean_wt_1':new_data_wt[:,:n//2].mean(axis=1),
                        'd1_median_wt_1':np.median(new_data_wt[:,:n//2],axis=1),
                        'd1_max_wt_1':new_data_wt[:,:n//2].max(axis=1),
                        'd1_std_wt_2':new_data_wt[:,n//2:].std(axis=1),
                        'd1_mean_wt_2':new_data_wt[:,n//2:].mean(axis=1),
                        'd1_median_wt_1':np.median(new_data_wt[:,n//2:],axis=1),
                        'd2_std':new_data_2.std(axis=1),
                        'd2_mean':new_data_2.mean(axis=1),
                        'd2_median':np.median(new_data_2,axis=1),
                        'd2_min':new_data_2.max(axis=1)-new_data_2.min(axis=1),
                        'd2_std_1':new_data_2[:,:n//2].std(axis=1),
                        'd2_mean_1':new_data_2[:,:n//2].mean(axis=1),
                        'd2_median_1':np.median(new_data_2[:,:n//2],axis=1),
                        'd2_max_1':new_data_2[:,:n//2].max(axis=1),
                        'd2_std_2':new_data_2[:,n//2:].std(axis=1),
                        'd2_mean_2':new_data_2[:,n//2:].mean(axis=1),
                        'd2_median_2':np.median(new_data_2[:,n//2:],axis=1),
                        'd2_std_wt':new_data_wt_2.std(axis=1),
                        'd2_mean_wt':new_data_wt_2.mean(axis=1),
                        'd2_median_wt':np.median(new_data_wt_2,axis=1),
                        'd2_min_wt':new_data_wt_2.max(axis=1)-new_data_wt_2.min(axis=1),
                        'd2_std_wt':new_data_wt_2.std(axis=1),
                        'd2_mean_wt_1':new_data_wt_2[:,:n//2].mean(axis=1),
                        'd2_median_wt_1':np.median(new_data_wt_2[:,:n//2],axis=1),
                        'd2_max_wt_1':new_data_wt_2[:,:n//2].max(axis=1),
                        'd2_std_wt_2':new_data_wt_2[:,n//2:].std(axis=1),
                        'd2_mean_wt_2':new_data_wt_2[:,n//2:].mean(axis=1),
                        'd2_median_wt_2':np.median(new_data_wt_2[:,n//2:],axis=1),
                        'slice_std_1':np.sum(np.std(lst_data_1.reshape(lst_data_1.shape[0], 10, -1), axis=2, ddof=1), axis=1),
                        'slice_std_2':np.sum(np.std(lst_data_2.reshape(lst_data_2.shape[0], 10, -1), axis=2, ddof=1), axis=1),
                        'slice_std_1_half': np.median(np.std(lst_data_1.reshape(lst_data_1.shape[0], 10, -1)[:,:5], axis=2, ddof=1), axis=1),
                        'slice_std_2_half': np.median(np.std(lst_data_2.reshape(lst_data_2.shape[0], 10, -1)[:,:5], axis=2, ddof=1), axis=1),
                        'slice_std_1_med': np.median(np.std(lst_data_1.reshape(lst_data_1.shape[0], 10, -1), axis=2, ddof=1), axis=1),
                        'slice_std_2_med': np.median(np.std(lst_data_2.reshape(lst_data_2.shape[0], 10, -1), axis=2, ddof=1), axis=1),
                        'slice_std_1_half_med': np.median(np.std(lst_data_1.reshape(lst_data_1.shape[0], 10, -1)[:,5:], axis=2, ddof=1), axis=1),
                        'slice_std_2_half_med': np.median(np.std(lst_data_2.reshape(lst_data_2.shape[0], 10, -1)[:,5:], axis=2, ddof=1), axis=1)})

    print('Идет нормализация данных')

    with open('normalize_max_border.json', 'r', encoding='utf-8') as f:
        max_border = json.load(f)
    with open('normalize_min_border.json', 'r', encoding='utf-8') as f:
        min_border = json.load(f)

    norm_lst = list(X.drop(columns=['Test_index', 'Presentation', 'Question']))

    is_normal_rasp = [  0,1,1,0,0,1,1,
                        1,0,1,1,1,0,1,
                        1,0,0,1,1,0,0,
                        1,0,1,1,0,0,1,
                        1,1,0,1,1,0,1,
                        1,0,1,1,0,0,1,
                        1,0,0,0,0,0,0,
                        0,0 ]

    normalize(X, min_border, max_border, norm_lst, is_normal_rasp)
    X = X.values

    model = CatBoostClassifier()
    model.load_model('cb_model_58.cbm', format='cbm')

    class_labels = model.predict(X)
    df['Class_label'] = class_labels

    #ВЫВОД ВСЕЙ ИНФОРМАЦИИ
    sg.theme('SystemDefaultForReal')
    data2 = df.loc[:, ['id', 'Class_label']]
    data2.to_csv('Marked_data.csv', sep=';', index=False)

    print('Размеченные данные успешно импортированы в файл Marked_data.xlsx')

    headings = list(data2)
    data2 = data2.values
    data2 = list(data2)
    data2 = list(map(list, data2))

    layout2 = [[sg.Table(values=data2, headings=headings, max_col_width=25,
                        auto_size_columns=False,
                        display_row_numbers=True,
                        justification='right',
                        num_rows=20,
                        alternating_row_color='lightblue',
                        key='-ROW-',
                        row_height=25,
                        tooltip='Это таблица')],
              [sg.Button('Посмотреть информацию')],
              [sg.Text('Вы можете посмотреть информацию об испытуемом нажав кнопку выше')]]

    window2 = sg.Window('Проведенные тесты', layout2)
    while True:
        event, values = window2.read()
        if event != None:
            k = np.array(values['-ROW-'])
            print(k)
        if event == sg.WIN_CLOSED:
            break
        #ГРАФИКИ
        graphs_vals_1 = lst_data_1[k].reshape(-1)
        graphs_vals_2 = lst_data_2[k].reshape(-1)
        
        class_lab = class_labels[k]
        
        c1 = ''
        if class_lab == 0:
            c1 = 'w'
        elif class_lab == 1:
            c1 = 'y'
        elif class_lab == 2:
            c1 = 'r'
        c2 = ''
        if class_lab == 0:
            c2 = 'w'
        elif class_lab == 1:
            c2 = 'y'
        elif class_lab == 2:
            c2 = 'r'

        assert c1 != ''
        assert c2 != ''
        
        fig, ax = plt.subplots(ncols=2, figsize=(18,6))
        
        ax[0].plot(graphs_vals_1, label='Фотоплетизмограмм')
        ax[0].axvspan(0, len(graphs_vals_1),  color=c1, alpha=0.3)
        ax[0].legend()
        ax[1].plot(graphs_vals_2, label='Пьезоплетизмограмма')
        ax[1].axvspan(0, len(graphs_vals_2),  color=c2, alpha=0.3)
        ax[1].legend()
       
        # matplotlib.use("TkAgg")

        def draw_figure(canvas, figure):
            figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
            figure_canvas_agg.draw()
            figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
            return figure_canvas_agg
        layout3 = [[sg.Text("Показания фотоплетизмограмма и пьезоплетизмограмма", font='Courier 14', text_color='Black')],
            [sg.Canvas(key="-CANVAS-")],
            [sg.Button("Ok")]]
        window3 = sg.Window(
            "Информация об испытуемом",
            layout3,
            location=(0, 0),
            finalize=True,
            element_justification="center",
            font="Helvetica 18",
        )
        draw_figure(window3["-CANVAS-"].TKCanvas, fig)

        event3, values3 = window3.read()
        window3.close() 

    window2.close()       
    
    
else:
    window1 = sg.Window("Работа скрипта", layout1)
    while True:
        event, values = window1.read()
        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break
        if event in (None, 'Exit', 'Cancel'):
            break
    window1.close()
window.close()