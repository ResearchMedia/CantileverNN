import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.pyplot import cm
from pathlib import Path
import numpy as np
import os

plot_file_ext = '.optim_plot'
root_path = Path('tmp/')
MAX_LINES_SHOWN = 1000
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
    xlim = (-5, 133)     # set the ylim to bottom, top
    ylim = (-5, 133)     # set the ylim to bottom, top
    axs_serial = axs.reshape(-1)
    for idx, file in enumerate(plot_files):
        ax = axs_serial[idx]   
        graph_data = open(file,'r').read()
        lines = graph_data.split('\n')
        # make sure last result has high contrast
        nr_samples = len(lines)-2 if len(lines)-2 > 0 else 0
        color_index = np.linspace(0.3,0.8,nr_samples)
        color_index = np.power(color_index,2)
        color_index = np.append(color_index, 1.0)
        cols = iter(cm.BuGn(color_index))
        ax.clear()
        ax.title.set_text(file.stem)        
        for i, line in enumerate(lines):
            if len(lines) > MAX_LINES_SHOWN and i < len(lines)-MAX_LINES_SHOWN and i > 1:
                continue
            xs = []
            ys = []
            if len(line) > 1:
                verts = line.split(')\",')
                for vert in verts:
                    #print("VERT:",vert)
                    x, y = vert.strip("\"").strip("()").split(", ")
                    #print("X:",x, "Y:", y)
                    xs.append(float(x))
                    ys.append(float(y))
                xs.append(xs[0])
                ys.append(ys[0])
                c = next(cols)
                ax.plot(xs, ys, color = c, linewidth=1.0)
                ax.scatter(xs, ys, color = c)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

def main():
    style.use('fivethirtyeight')
    global plot_files
    plot_files = get_plottable_files(root_path = root_path, plot_file_ext = plot_file_ext)
    print(plot_files)
    nr_plots = len(plot_files) if len(plot_files) > 1 else 2
    ncol = int(np.ceil(np.sqrt(nr_plots)))
    # size ncol to fit nr of graphs
    nrow = ncol if (ncol*(ncol-1)) < nr_plots else ncol-1
    global axs
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    #xlim = (-5, 133)     # set the ylim to bottom, top
    #ylim = (-2, 133)     # set the ylim to bottom, top
    #axs = np.array(axs)

    ani = animation.FuncAnimation(fig, animate, interval=100)
    #plt.setp(axs, xlim=xlim, ylim=ylim)
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

if __name__ == '__main__':
    main()
