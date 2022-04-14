import matplotlib.pyplot as plt


def plot_pva(data, ref):
    ylabel = ['lat/deg', 'lot/deg', 'altitude/m', 'Vx/m/s', 'Vy/m/s', 'Vz/m/s', 'roll/deg', 'pitch/deg', 'yaw/deg']
    title = ['lat_compare', 'lat_compare', 'altitude_compare', 'Vx_compare', 'Vy_compare', 'Vz_compare', 'roll_compare',
             'pitch_compare', 'yaw_compare']
    for i in range(9):
        fig_i, ax = plt.subplots(figsize=(12, 8))
        ax.plot(data[:, 1], data[:, i + 2], 'b', label='fusion')
        ax.plot(ref[:, 1], ref[:, i + 2], 'r', label='ref')

        ax.legend(loc=2)
        ax.set_xlabel('time(s)')
        ax.set_ylabel(ylabel[i])
        ax.set_title(title[i])
        plt.grid()
    plt.show()


def plot_pva_error(time, error):
    line_label = ['N', 'E', 'D', 'N', 'E', 'D', 'roll', 'pitch', 'yaw']
    ylabel = ['error(m)', 'error(m/s)', 'error(deg)']
    title = ['positon_error', 'velocity_error', 'attitude_error']
    for i in range(3):
        fig_i, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time, error[:, 3 * i], 'r', label=line_label[3 * i])
        ax.plot(time, error[:, 3 * i + 1], 'g', label=line_label[3 * i + 1])
        ax.plot(time, error[:, 3 * i + 2], 'b', label=line_label[3 * i + 2])
        ax.legend(loc=2)
        ax.set_xlabel('time(s)')
        ax.set_ylabel(ylabel[i])
        ax.set_title(title[i])
        plt.grid()
    plt.show()
