import numpy as np
import joblib
import json
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
import argparse
from matplotlib import pylab as plt
from matplotlib import animation


def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)


class PlotFig:
    def draw_heatmap(self, h1, cmap, vmin=-1, vmax=1, relative=True):
        if relative:
            im = plt.imshow(
                h1, aspect="auto", interpolation="none", cmap=cmap, origin="lower"
            )
            plt.colorbar(im)
        else:
            plt.imshow(
                h1,
                aspect="auto",
                interpolation="none",
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")

    # line: num x steps
    def draw_line(self, line, label, color, num, alpha=1.0, pred=False):
        if pred:
            plt.plot(
                np.concatenate([[np.nan], line[0, :]]),
                label=label,
                color=color,
                alpha=alpha,
            )
            for i in range(num - 1):
                plt.plot(
                    np.concatenate([[np.nan], line[i + 1, :]]), color=color, alpha=alpha
                )
        else:
            plt.plot(line[0, :], label=label, color=color, alpha=alpha)
            for i in range(num - 1):
                plt.plot(line[i + 1, :], color=color, alpha=alpha)

    def draw_scatter_z(self, x, y, c, alpha=1.0, axis_x=0, axis_y=1):
        plt.scatter(x, y, c=c, alpha=alpha)
        min_x, max_x = self.min_z[axis_x], self.max_z[axis_x]
        offset_x = (max_x - min_x) * 0.01
        plt.xlim(min_x - offset_x, max_x + offset_x)
        min_y, max_y = self.min_z[axis_y], self.max_z[axis_y]
        offset_y = (max_y - min_y) * 0.01
        plt.ylim(min_y - offset_y, max_y + offset_y)

    def build_data(self, args, data, result_data):
        self.colorlist = ["g", "b", "r", "c", "m", "y", "k", "w"]
        self.cmap_z = generate_cmap(["#0000FF", "#FFFFFF", "#FF0000"])
        self.cmap_x = generate_cmap(["#FFFFFF", "#000000"])
        # z
        self.z_q = None
        self.mu_q = None
        self.sample_z = None
        if "z_q" in result_data:
            # deprecated
            self.z_q = result_data["z_q"]
            if "mu_q" in result_data:
                self.mu_q = result_data["mu_q"]
                print("z: z_q, mu_q")
            else:
                print("z: z_q")
        elif "z_params" in result_data:
            self.z_q = result_data["z_params"][0]
            print("z: z_params")
        elif "z" in result_data:
            # sampling
            self.z_q = np.mean(result_data["z"], axis=0)
            self.sample_z = result_data["z"]
            print("z: z")
        else:
            print("z: nothing")
        if self.z_q is not None:
            print("z: ", self.z_q.shape)
        # recons
        self.obs_mu = None
        self.sample_obs = None
        if "obs_params" in result_data:
            self.obs_mu = result_data["obs_params"][0]
            print("reconstructed obs:", self.obs_mu.shape)
        elif "mu" in result_data:
            # sampling
            self.obs_mu = np.mean(result_data["mu"], axis=0)
            self.sample_obs = result_data["mu"]
            print("reconstructed obs:", self.obs_mu.shape)
        else:
            print("reconse: nothing")
        print("obs:", data.x.shape)
        # pred_mu=result_data["pred_params"][0]
        if "obs_pred_params" in result_data:
            self.pred_mu = result_data["obs_pred_params"][0]
        if data.mask is not None and np.any(data.mask < 0.1):
            data.x[data.mask < 0.1] = np.nan
            if self.obs_mu:
                self.obs_mu[data.mask < 0.1] = np.nan
            if self.pred_mu:
                self.pred_mu[data.mask < 0.1] = np.nan

        sample_num_z = 10000
        self.all_z = np.reshape(self.z_q[:, :, :], (-1, self.z_q.shape[2]))
        self.max_z = np.max(self.all_z, axis=0)
        self.min_z = np.min(self.all_z, axis=0)
        print(self.max_z)
        print(self.min_z)
        idx = np.random.randint(len(self.all_z), size=sample_num_z)
        self.all_z = self.all_z[idx, :]

    def plot_z(self, args, idx, s, z):
        colorlist_z = self.colorlist
        z_plot_type = args.z_plot_type
        if z_plot_type == "line":
            plot_dim = min([len(colorlist_z), z.shape[1]])
            for i in range(plot_dim):
                plt.plot(z[:, i], label="dim-" + str(i) + "/" + str(z.shape[1]))

        elif z_plot_type == "heatmap":
            # h=mu_q[idx,:s,:]
            print("upper plot:z_q:", z.shape)
            self.draw_heatmap(np.transpose(z), self.cmap_z)
        else:  # scatter
            self.draw_scatter_z(self.all_z[:, 0], self.all_z[:, 1], c="b", alpha=0.1)
            if args.anim:
                plt.plot(z[:, 0], z[:, 1], c="r", alpha=0.5)
            else:
                self.draw_scatter_z(z[:, 0], z[:, 1], c="r", alpha=0.8)

        plt.legend()
        plt.title("z: state space")

    def plot_filter_z(self, args, idx, s, z):
        colorlist_z = self.colorlist
        z_plot_type = args.z_plot_type
        if z_plot_type == "line":
            plot_dim = min([len(colorlist_z), z.shape[1]])
            for i in range(plot_dim):
                # plt.plot(z[:, i], label="dim-"+str(i)+"/"+str(z.shape[1]))
                label = "dim-" + str(i) + "/" + str(z.shape[1])
                self.draw_line(
                    self.sample_z[:, idx, :s, i],
                    label=label,
                    color=colorlist_z[i],
                    num=args.num_particle,
                )
            plt.legend()
            plt.title("z: state space")
        else:  # scatter
            self.draw_scatter_z(self.all_z[:, 0], self.all_z[:, 1], c="b", alpha=0.1)
            self.draw_scatter_z(z[:, 0], z[:, 1], c="r", alpha=0.8)
            plt.legend()
            plt.title("z: state space")

    def plot_x(self, args, idx, s, data):
        cmap_x = self.cmap_x
        d = args.obs_dim
        x_plot_type = args.x_plot_type
        if x_plot_type == "line":
            plt.plot(data.x[idx, :s, d], label="x")
            plt.plot(self.obs_mu[idx, :s, d], label="recons")
            plt.plot(np.concatenate([[np.nan], self.pred_mu[idx, :s, d]]), label="pred")
        else:
            self.draw_heatmap(np.transpose(data.x[idx, :s, :]), cmap_x)
        plt.legend()
        plt.title("x: observation")

    def plot_filter_x(self, args, idx, s, data):
        cmap_x = self.cmap_x
        d = args.obs_dim
        x_plot_type = args.x_plot_type
        if x_plot_type == "line":
            self.draw_line(
                self.sample_obs[:, idx, :s, d],
                label="pred",
                color="g",
                num=args.obs_num_particle,
                alpha=0.5,
                pred=True,
            )
            plt.plot(
                np.concatenate([[np.nan], self.obs_mu[idx, :s, d]]),
                label="pred_mean",
                color="r",
            )
            plt.plot(data.x[idx, :s, d], label="x", color="b")
        else:
            self.draw_heatmap(np.transpose(data.x[idx, :s, :]), cmap_x)
        plt.legend()
        plt.title("x: observation")

    def plot_fig(self, idx, args, data, result_data):
        colorlist = self.colorlist
        colorlist_z = self.colorlist
        cmap_z = self.cmap_z
        cmap_x = self.cmap_x
        d = args.obs_dim
        if data.s is not None:
            s = data.s[idx]
        else:
            s = data.x.shape[1]
        print("data index:", idx)
        # print("data =",data.info[data.pid_key][idx])
        if self.obs_mu is not None:
            error = (self.obs_mu[idx, :-1, :] - data.x[idx, 1:, :]) ** 2
            print("error=", np.nanmean(error))
        print("dimension (observation):", d)
        print("steps:", s)

        fig = plt.figure()
        if args.mode == "data":
            plt.subplot(1, 1, 1)
            self.draw_heatmap(np.transpose(data.x[idx, :s, :]), cmap_x)
            plt.legend()
            plt.title("x: observation")

        elif args.mode == "filter":
            plt.subplot(3, 1, 1)
            z = self.z_q[idx, :s, :]
            self.plot_filter_z(args, idx, s, z)

            plt.subplot(3, 1, 2)
            self.plot_filter_x(args, idx, s, data)

            plt.subplot(3, 1, 3)
            e = (self.sample_obs[:, idx, :s, d] - data.x[idx, :s, d]) ** 2
            self.draw_line(e, label="error", color="b", num=args.obs_num_particle)
            plt.legend()
            plt.title("error")
        else:

            plt.subplot(2, 1, 1)
            z = self.z_q[idx, :s, :]
            self.plot_z(args, idx, s, z)
            if args.anim:
                im1 = plt.plot([], [], c="r", marker="o")

            plt.subplot(2, 1, 2)
            self.plot_x(args, idx, s, data)
            if args.anim:
                h = data.x.shape[2]
                if args.x_plot_type == "line":
                    im2 = plt.plot([0, 0], [0, h], c="r", marker="o")
                else:
                    im2 = plt.plot([0, 0], [0, h], c="r")

        if args.anim:

            def animate(t, im1, im2):
                im1[0].set_data([z[t, 0]], [z[t, 1]])
                im2[0].set_data([t, t], [0, h])
                return im1[0], im2[0]

            anim = animation.FuncAnimation(
                fig, animate, fargs=(im1, im2), frames=s, interval=500, blit=True
            )
            return anim
        return None


def plot_start():
    parser = get_default_argparser()
    parser.add_argument(
        "--obs_dim", type=int, default=0, help="a dimension of observation for plotting"
    )
    parser.add_argument(
        "--num_particle",
        type=int,
        default=10,
        help="the number of particles (latent variable)",
    )
    parser.add_argument(
        "--obs_num_particle",
        type=int,
        default=10,
        help="the number of particles (observation)",
    )
    parser.add_argument(
        "--x_plot_type",
        type=str,
        default="line",
        help="plot type for observation",
        choices=["line", "heatmap", "scatter"],
    )
    parser.add_argument(
        "--z_plot_type",
        type=str,
        default="line",
        help="plot type for state space",
        choices=["line", "heatmap", "scatter"],
    )
    parser.add_argument(
        "--anim", action="store_true", default=False, help="Animation",
    )

    args = parser.parse_args()
    # config
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pylab as plt
    else:
        from matplotlib import pylab as plt
    ##
    print("=============")
    print("... loading")
    # d=data_info["attr_emit_list"].index("206010")
    # print("206010:",d)
    if args.config is None:
        parser.print_help()
        quit()
    data, result_data = load_plot_data(args)
    print("data:", data.keys())
    print("result:", result_data.keys())
    print("=============")
    plotter = PlotFig()
    plotter.build_data(args, data, result_data)
    if args.index is None:
        l = len(result_data.info[result_data.pid_key])
        if args.limit_all is not None and l > args.limit_all:
            l = args.limit_all
        for idx in range(l):
            anim = plotter.plot_fig(idx, args, data, result_data)
            name = result_data.info[result_data.pid_key][idx]
            if args.anim:
                out_filename = (
                    result_data.out_dir
                    + "/"
                    + str(idx)
                    + "_"
                    + name
                    + "_"
                    + args.mode
                    + ".mp4"
                )
                print("[SAVE] :", out_filename)
                anim.save(out_filename, fps=10, extra_args=["-vcodec", "libx264"])
            else:
                out_filename = (
                    result_data.out_dir
                    + "/"
                    + str(idx)
                    + "_"
                    + name
                    + "_"
                    + args.mode
                    + ".png"
                )
                print("[SAVE] :", out_filename)
                plt.savefig(out_filename)
                plt.clf()
    else:
        idx = args.index
        anim = plotter.plot_fig(idx, args, data, result_data)
        name = result_data.info[result_data.pid_key][idx]
        if args.anim:
            out_filename = (
                result_data.out_dir
                + "/"
                + str(idx)
                + "_"
                + name
                + "_"
                + args.mode
                + ".mp4"
            )
            print("[SAVE] :", out_filename)
            anim.save(out_filename, fps=10, extra_args=["-vcodec", "libx264"])
        else:
            out_filename = (
                result_data.out_dir
                + "/"
                + str(idx)
                + "_"
                + name
                + "_"
                + args.mode
                + ".png"
            )
            print("[SAVE] :", out_filename)
            plt.savefig(out_filename)
            plt.show()
            plt.clf()

def plot_field(config,args):
    filename=config["result_path"]+"/sim/field_pt.npy"
    if os.path.exists(filename):
        print("[LOAD]", filename)
        pt = np.load(filename)
        filename=config["result_path"]+"/sim/field_vec.npy"
        if os.path.exists(filename):
            vec = np.load(filename)
            if vec.shape[1]==1:
                """
                plt.quiver(pt[:,0],0,vec[:,0],-1)
                """
                x=data_pt[:,0]
                y=data_pt[:,0]+data_vec[:,0]
                fx=(y-x)
                plt.plot(x,fx)
                plt.xlabel("x")
                plt.ylabel("f(x)")
            else:
                plt.quiver(pt[:,0],pt[:,1],vec[:,0],vec[:,1])
            plt.title("state-space")
            plt.grid()
            plt.legend()
            filename=config["result_path"]+"/field.png"
            print("[SAVE]",filename)
            plt.savefig(filename)


def main():
    # plot_start()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random", type=int, default=10, help="config json file"
    )
    parser.add_argument(
            "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument("--no-config", action="store_true", help="use default setting")
    args = parser.parse_args()
    
    result_path="result"
    ## config
    if args.config:
        config=json.load(open(args.config))
        result_path=config["result_path"]
    
    filename=result_path+"/sim/input.npy"
    print("[LOAD]",filename)
    data_u = np.load(filename)
    filename=result_path+"/sim/obs.npy"
    print("[LOAD]",filename)
    data_x = np.load(filename)
    filename=result_path+"/sim/obs_gen.npy"
    print("[LOAD]",filename)
    data_xg = np.load(filename)
    filename=result_path+"/sim/states.npy"
    print("[LOAD]",filename)
    data_z = np.load(filename)
    filename=result_path+"/sim/state_true.npy"
    print("[LOAD]",filename)
    data_z_true = np.load(filename)
    n=data_x.shape[0]
    ##
    print("u       :",data_u.shape)
    print("x       :",data_x.shape)
    print("x_recons:",data_xg.shape)
    print("z       :",data_z.shape)
    print("z_true  :",data_z_true.shape)
    print("#data   :",n)

    ##
    plot_field(config,args)
    # plot_start()
    idx_all=np.arange(n)
    np.random.shuffle(idx_all)
    for idx in idx_all[:args.random]:
        print("### index =",idx)
        z = data_z[idx, :, :]
        z_true = data_z_true[idx, :, :]
        u = data_u[idx, :, :]
        x = data_x[idx, :, :]
        xg = data_xg[idx, :, :]
        fig = plt.figure()
        plt.subplot(4, 1, 1)
        plot_dim = u.shape[1]
        for i in range(plot_dim):
            plt.plot(u[:, i], label="u dim-" + str(i) + "/" + str(u.shape[1]))
        plt.title("input")
        plt.legend()
        plt.subplot(4, 1, 2)
        plot_dim = z.shape[1]
        for i in range(plot_dim):
            plt.plot(z[:, i], label="dim-" + str(i) + "/" + str(z.shape[1]))
        plt.title("state")
        plt.legend()
        plt.subplot(4, 1, 3)
        plot_dim = z_true.shape[1]
        for i in range(plot_dim):
            plt.plot(z_true[:, i], label="dim-" + str(i) + "/" + str(z_true.shape[1]))
        plt.title("true state")
        plt.legend()

        plt.subplot(4, 1, 4)
        plot_dim = x.shape[1]
        for i in range(plot_dim):
            plt.plot(x[:, i], label="x dim-" + str(i) + "/" + str(x.shape[1]))
        plot_dim = xg.shape[1]
        for i in range(plot_dim):
            plt.plot(xg[:, i], label="recons dim-" + str(i) + "/" + str(x.shape[1]))
        plt.title("observation")
        plt.legend()
        filename=config["result_path"]+"/fig{:04d}.png".format(idx)
        print("[SAVE]",filename)
        plt.savefig(filename)


if __name__ == "__main__":
    main()
