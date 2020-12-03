import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.pyplot import cm
from pathlib import Path
import numpy as np
import os, csv

plot_file_ext = '.age_pareto'
root_path = Path('/home/ron/Downloads/optimized_beam_test/Evolution/')
MAX_EPOCH = 1000
MAPE_INVERSION = True 
axs = None
plot_files = None
def get_plottable_files(root_path = None, plot_file_ext = None):
    plottable_files = []
    for file in os.listdir(root_path):
        if file.endswith(plot_file_ext):
            plottable_files.append(Path(os.path.join(root_path, file)))
    plottable_files.sort()
    return plottable_files


def animate(i):
    global MAX_EPOCH
    nr_lines = 0
    axs_serial = axs.reshape(-1)
    for idx, file in enumerate(plot_files):
        ax = axs_serial[2*idx]
        ax_age = axs_serial[2*idx+1]
        with open(file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            # make sure last result has high contrast 
            color_index = np.linspace(0,1,MAX_EPOCH)
            color_index = np.append(color_index, 1.0)
            cols = cm.rainbow(color_index)
            ax.clear()
            ax.set_xlabel('epoch')
            ax_age.set_xlabel('age')
            if MAPE_INVERSION:
                ax.set_ylabel('1-MAPE')
                ax_age.set_ylabel('1-MAPE')
            else:
                ax.set_ylabel('error')
                ax_age.set_ylabel('error')

            ax.title.set_text(file.stem)
            prev_epoch = 0
            xs = []
            ys = []
            x_age = []
            best_ages = []
            best_epochs = []
            for i, line in enumerate(reader):
                if len(line) > 1:
                    xs.append(float(line['epoch']))
                    x_age.append(float(line['age']))
                    if MAPE_INVERSION:
                        ys.append(1-float(line['fitness']))
                    else:
                        ys.append(float(line['fitness']))
                if (prev_epoch < int(line['epoch'])):
                    c = cols[(int(line['epoch'])-1)%MAX_EPOCH]
                    ax.scatter(xs, ys, color = c, marker='.')
                    ax_age.scatter(x_age, ys, color = c, marker='.')
                    xs = []
                    ys = []
                    x_age = []
                    prev_epoch = int(line['epoch'])
                    nr_lines += 1
    MAX_EPOCH = int(nr_lines*1.1)

def main():
    style.use('fivethirtyeight')
    global plot_files
    plot_files = get_plottable_files(root_path = root_path, plot_file_ext = plot_file_ext)
    print(plot_files)
    nr_plots = len(plot_files*2) 
    ncol = int(np.ceil(np.sqrt(nr_plots)))
    # size ncol to fit nr of graphs
    nrow = ncol if (ncol*(ncol-1)) < nr_plots else ncol-1
    global axs
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    #xlim = (-5, 133)     # set the ylim to bottom, top
    #ylim = (-2, 133)     # set the ylim to bottom, top
    #axs = np.array(axs)

    ani = animation.FuncAnimation(fig, animate, interval=500)
    #plt.setp(axs, xlim=xlim, ylim=ylim)
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

if __name__ == '__main__':
    main()
