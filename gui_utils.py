import numpy as np
from numpy.typing import NDArray
import os
from typing import List, Dict, Set
from pandas import read_csv, DataFrame, Series
from ipywidgets import (
    Checkbox,
    Button,
    FloatSlider,
    Layout,
    HBox,
    VBox,
    GridBox,
    Output,
    Label,
    IntProgress,
)
from IPython.display import display
from scipy.stats import iqr
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.graph_objects import Figure, Scatterpolar, FigureWidget, Scatter

NDArrayFloat = NDArray[np.float_]
NDArrayInt = NDArray[np.int_]
home_dir = os.getcwd()



class MultiObjectiveGUI:
    if "/" in __file__:
        path_str: str = __file__.rsplit("/", 1)[0] + "/"
    else:
        path_str: str = "/"
    csv_filename: str = path_str + "/material_data.csv"

    def __init__(self, allowed_metric_list: List[str], dataframe_info: DataFrame):
        # Data frames to store data
        self.allowed_metrics: List[str] = allowed_metric_list
        self.df_orig: DataFrame = dataframe_info.copy(deep=True)
        self.filtered_df: DataFrame = dataframe_info.copy(deep=True)
        self.pareto_df: DataFrame = dataframe_info.copy(deep=True)

        # Filtering data
        self.selected_metrics: List[str] = []
        self.metric_cutoffs: List[float] = []
        self.num_filtered: List[float] = []
        self.filter_class: List[str] = []
        self.filter_cols: List[str] = []
        self.starting_ds_count: int = self.df_orig.shape[0]
        self.filtered_count: int = 0

        # Pareto Front Identification
        self.pareto_metrics: List[str] = []
        self.metric_weights: List[float] = []
        self.X_for_pareto = None
        self.err_bars = None
        self.ps = None
        self.nps = None
        self.paretomeans = None
        self.paretostds = None
        self.norm_err_bars = None
        # Other useful stuff
        self.classify_metrics = [
            "radioactive",
            "toxic",
            "f_block",
            "more_than_four_elems",
        ]
        self.cl_chbox_description = [
            "Do not include radioactive elements",
            "Do not include potentially toxic elements",
            "Do not include f_block elements",
            "Do not include materials with more than 4 elements",
        ]
        self.less_than_metrics = ["e_above_hull", "nsites"]

        # Widgets
        self.title: Button = Button(description="Top Interconnect Candidate Identifier")
        self.gui_out: Output = Output()
        self.output_plot: Output = Output()
        self.fig_widget = None
        self.output_plot_2: Output = Output()
        self.fig_pareto = None

        self.checkbox_metrics: List[Checkbox] = []
        self.checkbox_pareto: List[Checkbox] = []
        self.checkbox_add_to_plot: List[Checkbox] = []
        self.benchmark_mat = ["mp-30", "mp-33", "mp-54", "mp-23"] # "mp-219", "mp-7577", 
        #                      "mp-19210", "mp-998", "mp-12108", "mp-2078",
        #                      "mp-11531", "mp-1077077", "mp-12794", "mp-22746"

        self.select_metrics_button = Button(
            description="Select Metrics",
            disabled=False,
            tooltip="Click to select metrics",
        )
        self.select_metrics_button.on_click(self.select_metrics)

        self.filter_cutoffs_button = Button(
            description="Filter Materials by Cutoffs",
            disabled=True,
            tooltip="Click to filter material lists by chosen cutoffs",
        )
        self.filter_cutoffs_button.on_click(self.filter_by_cutoffs)

        self.find_paerto_button = Button(
            description="Identify Pareto optimal candidates",
            disabled=True,
            tooltip="Click to identify the pareto optimal candidates",
        )
        self.find_paerto_button.on_click(self.select_pareto)

        self.plot_top_cand_button = Button(
            description="Plot Top Candidates candidates",
            disabled=True,
            tooltip="Click to display the top interconnect candidates",
        )
        self.plot_top_cand_button.on_click(self.compute_objectives)

        self.add_to_plot = Button(
            description="Add Benchmark Materials",
            disabled=True,
            tooltip="Add benchmark materials to plot",
        )
        self.add_to_plot.on_click(self.add_benchmarks)

        self.fs_style = {"description_width": "initial"}
        self.fs_layout = Layout(height="auto", width="auto")
        self.cutoff_sliders: List[FloatSlider] = []
        self.weight_sliders: List[FloatSlider] = []
        self.alpha_slider: FloatSlider = FloatSlider(
            value=1,
            min=0,
            max=10,
            step=0.01,
            description="Regularization Constant",
            disabled=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=self.fs_style,
            layout=self.fs_layout,
        )
        self.filtered_count_Label: List[Label] = []
        self.pareto_anal_ct_label = None
        self.pareto_sampling_progress: IntProgress = IntProgress(
            min=-1,
            max=10,
            value=0,
            description="Pareto Sampling Progress",
            style=self.fs_style,
        )
        self.pareto_ct_label = None

        for metric in self.allowed_metrics:
            if metric in self.classify_metrics:
                chbox: Checkbox = Checkbox(
                    value=True,
                    description=self.cl_chbox_description[
                        self.classify_metrics.index(metric)
                    ],
                    disabled=False,
                    style=self.fs_style,
                )
            else:
                chbox: Checkbox = Checkbox(
                    value=True, description=metric, disabled=False, style=self.fs_style
                )
            self.checkbox_metrics.append(chbox)

        self.df_benchmark: DataFrame = self.df_orig.loc[
            self.df_orig["task_id"].isin(self.benchmark_mat)
        ]
        self.df_benchmark.reset_index(drop=True, inplace=True)
        n_bm = len(self.benchmark_mat)
        for i_bm in range(0, n_bm):
            chbox: Checkbox = Checkbox(
                value=True,
                description=self.df_benchmark.at[i_bm, "pretty_formula"],
                disabled=False,
                style=self.fs_style,
            )
            self.checkbox_add_to_plot.append(chbox)

        display(self.gui_out)
        self.display_all()

    def select_metrics(self, button_instance):

        # Reinitialize everything

        self.filtered_df: DataFrame = self.df_orig.copy(deep=True)
        self.pareto_df: DataFrame = self.df_orig.copy(deep=True)

        # Filtering data
        self.selected_metrics: List[str] = []
        self.metric_cutoffs: List[float] = []
        self.num_filtered: List[float] = []
        self.filter_class: List[str] = []
        self.filter_cols: List[str] = []
        self.starting_ds_count: int = self.df_orig.shape[0]
        self.filtered_count: int = 0

        # Pareto Front Identification
        self.pareto_metrics: List[str] = []
        self.metric_weights: List[float] = []
        self.X_for_pareto = None
        self.err_bars = None
        self.ps = None
        self.nps = None
        self.paretomeans = None
        self.paretostds = None

        # Widgets
        self.cutoff_sliders: List[FloatSlider] = []
        self.weight_sliders: List[FloatSlider] = []
        self.pareto_sampling_progress: IntProgress = IntProgress(
            min=-1,
            max=10,
            value=0,
            description="Pareto Sampling Progress",
            style=self.fs_style,
        )
        self.checkbox_pareto: List[Checkbox] = []
        self.filtered_count_Label: List[Label] = []
        self.pareto_anal_ct_label = None
        self.pareto_ct_label = None
        self.output_plot: Output = Output()
        self.output_plot_2: Output = Output()
        self.fig_pareto = None

        for chbox in self.checkbox_metrics:
            if chbox.value == 1:
                if chbox.description in self.cl_chbox_description:
                    metric = self.classify_metrics[
                        self.cl_chbox_description.index(chbox.description)
                    ]
                    self.filter_class.append(metric)
                    self.starting_ds_count = self.starting_ds_count - np.sum(
                        self.filtered_df[metric].to_numpy()
                    )
                else:
                    metric = chbox.description
                    self.selected_metrics.append(metric)
                    if metric == "nsites":
                        values: NDArrayInt = self.df_orig[metric].to_numpy()
                        maxima = np.max(values)
                        minima = np.min(values)
                        fs = FloatSlider(
                            value=100,
                            min=minima,
                            max=maxima,
                            step=1,
                            description="Cutoff for {}".format(metric),
                            disabled=False,
                            orientation="horizontal",
                            readout=True,
                            readout_format=".0f",
                            style=self.fs_style,
                            layout=self.fs_layout,
                        )
                        filt_label = Label(
                            value="Number of Materials Filtered by metric will be displayed here",
                            layout=self.fs_layout,
                        )
                    else:
                        values: NDArrayFloat = self.df_orig[metric].to_numpy()
                        maxima = np.max(values)
                        minima = np.min(values)
                        step = np.std(values) / 100
                        fs = FloatSlider(
                            value=np.mean(values),
                            min=minima,
                            max=maxima,
                            step=step,
                            description="Cutoff for {}".format(metric),
                            disabled=False,
                            orientation="horizontal",
                            readout=True,
                            readout_format=".4f",
                            style=self.fs_style,
                            layout=self.fs_layout,
                        )
                        filt_label = Label(
                            value="Number of Materials Filtered by metric will be displayed here",
                            layout=self.fs_layout,
                        )
                    self.filtered_count_Label.append(filt_label)
                    self.cutoff_sliders.append(fs)

        self.filter_cutoffs_button.disabled = False
        self.display_all()

    def filter_by_cutoffs(self, button_instance):
        # Reinitialize appropriate elems:
        self.filtered_df: DataFrame = self.df_orig.copy()
        self.pareto_df: DataFrame = self.filtered_df.copy(deep=True)

        self.weight_sliders: List[FloatSlider] = []
        self.checkbox_pareto: List[Checkbox] = []
        self.filtered_count_Label: List[Label] = []
        self.pareto_sampling_progress: IntProgress = IntProgress(
            min=-1,
            max=10,
            value=0,
            description="Pareto Sampling Progress",
            style=self.fs_style,
        )
        self.pareto_anal_ct_label = None
        self.pareto_ct_label = None
        self.output_plot: Output = Output()
        self.output_plot_2: Output = Output()
        self.fig_pareto = None

        self.filter_cols: List[str] = []

        # Pareto Front Identification
        self.pareto_metrics: List[str] = []
        self.metric_weights: List[float] = []
        self.X_for_pareto = None
        self.err_bars = None
        self.ps = None
        self.nps = None
        self.paretomeans = None
        self.paretostds = None

        for col in self.filter_class:
            self.filtered_df = self.filtered_df[self.filtered_df[col] == 0]

        self.filtered_df.reset_index(drop=True, inplace=True)

        n_metric: int = len(self.selected_metrics)
        for i_metric in range(0, n_metric):
            metric: str = self.selected_metrics[i_metric]
            slider: FloatSlider = self.cutoff_sliders[i_metric]
            cutoff: float = slider.value
            self.metric_cutoffs.append(cutoff)
            fc_name: str = "{}_filter".format(metric)
            self.filter_cols.append(fc_name)
            self.filtered_df[fc_name] = False
            if metric in self.less_than_metrics:
                self.filtered_df[fc_name] = Series(self.filtered_df[metric] < cutoff)
            else:
                self.filtered_df[fc_name] = Series(self.filtered_df[metric] > cutoff)
            n_filt = np.sum(self.filtered_df[fc_name].to_numpy())
            self.num_filtered.append(n_filt)
            filt_label = Label(
                value="{} remaining out of {}".format(
                    n_filt, self.filtered_df.shape[0]
                ),
                layout=self.fs_layout,
            )
            self.filtered_count_Label.append(filt_label)

            chbox: Checkbox = Checkbox(
                value=True, description="{}".format(metric), disabled=False
            )
            self.checkbox_pareto.append(chbox)

        for col in self.filter_cols:
            self.filtered_df = self.filtered_df[self.filtered_df[col] == 1]

        self.filtered_df.drop(self.filter_cols, axis="columns")
        self.filtered_df.reset_index(drop=True, inplace=True)
        self.pareto_df = self.filtered_df.copy()

        self.pareto_anal_ct_label = Label(
            value="After all filters: {} Materials For Pareto "
            "Front Search Considered".format(self.pareto_df.shape[0]),
            layout=self.fs_layout,
        )
        self.pareto_sampling_progress.min = 0
        self.pareto_sampling_progress.max = self.pareto_df.shape[0]
        self.find_paerto_button.disabled = False
        self.display_all()

    @staticmethod
    def obtain_discrete_pareto_optima(
        x_moo: NDArrayFloat,
        error_bars: NDArrayFloat = None,
        use_errors: bool = True,
        report_prog: bool = True,
        prog_bar: IntProgress = None,
    ):
        pareto_space: Set[int] = set(np.argmax(x_moo, axis=0))
        non_pareto_space: Set[int] = set()

        n_points, n_obj = x_moo.shape
        if not use_errors or error_bars is None:
            error_bars: NDArrayFloat = np.zeros((n_points, n_obj))

        for i_dp in range(0, n_points):
            if report_prog and prog_bar is not None:
                prog_bar.value += 1

            set_dp = {i_dp}
            sol_i = x_moo[i_dp] + error_bars[i_dp]
            pareto_flag = True
            remove_elems = set()
            for elem in pareto_space:
                set_par = {elem}
                if np.all(sol_i > x_moo[elem] - error_bars[elem]):
                    if np.all(
                        sol_i - 2 * error_bars[elem] > x_moo[elem] + error_bars[elem]
                    ):
                        non_pareto_space = non_pareto_space.union(set_par)
                        remove_elems = remove_elems.union(set_par)
                elif np.all(sol_i < x_moo[elem] - error_bars[elem]):
                    non_pareto_space = non_pareto_space.union(set_dp)
                    pareto_flag = False
                    break
            pareto_space = pareto_space - remove_elems
            if pareto_flag:
                pareto_space = pareto_space.union(set_dp)
        if (
            len(pareto_space.intersection(non_pareto_space)) > 0
            and len(pareto_space.union(non_pareto_space)) != n_points
        ):
            print("Some issue")

        return list(pareto_space), list(non_pareto_space)

    def select_pareto(self, button_instance):
        # Reinitialize appropriate elems:
        self.pareto_df: DataFrame = self.filtered_df.copy(deep=True)

        self.weight_sliders: List[FloatSlider] = []
        self.output_plot: Output = Output()
        self.output_plot_2: Output = Output()
        self.fig_pareto = None

        # Pareto Front Identification
        self.pareto_metrics: List[str] = []
        self.metric_weights: List[float] = []
        self.X_for_pareto = None
        self.err_bars = None
        self.ps = None
        self.nps = None
        self.paretomeans = None
        self.paretostds = None

        self.X_for_plot = None
        self.plotmeans = None
        self.plotstds = None

        for chbox in self.checkbox_pareto:
            if chbox.value == 1:
                metric = chbox.description
                self.pareto_metrics.append(metric)
                fs = FloatSlider(
                    value=0,
                    min=0,
                    max=1,
                    step=0.01,
                    description="Weight for {}".format(metric),
                    disabled=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format=".2f",
                    style=self.fs_style,
                    layout=self.fs_layout,
                )
                self.weight_sliders.append(fs)

        self.X_for_pareto = self.pareto_df[self.pareto_metrics].to_numpy()
        n_dp, n_met = self.X_for_pareto.shape
        self.err_bars = np.zeros(self.X_for_pareto.shape)

        for i_met in range(0, n_met):
            if self.pareto_metrics[i_met] == "vol_mag":
                self.err_bars[:, i_met] = (
                    iqr(self.X_for_pareto[:, i_met]) * np.ones(n_dp) / 5
                )
            elif self.pareto_metrics[i_met] == "Norm DoS":
                self.err_bars[:, i_met] = (
                    iqr(self.X_for_pareto[:, i_met]) * np.ones(n_dp) / 5
                )
            elif self.pareto_metrics[i_met] == "vf":
                self.err_bars[:, i_met] = (
                    iqr(self.X_for_pareto[:, i_met]) * np.ones(n_dp) / 5
                )
            else:
                self.err_bars[:, i_met] = (
                    iqr(self.X_for_pareto[:, i_met]) * np.ones(n_dp) / 10
                )

        self.ps, self.nps = MultiObjectiveGUI.obtain_discrete_pareto_optima(
            x_moo=self.X_for_pareto,
            error_bars=self.err_bars,
            use_errors=True,
            prog_bar=self.pareto_sampling_progress,
            report_prog=True,
        )

        self.fig_pareto = FigureWidget()
        pca: PCA = PCA(n_components=2)
        x_plot = pca.fit_transform(X=self.X_for_pareto)

        scat = Scatter(
            x=x_plot[self.ps, 0],
            y=x_plot[self.ps, 1],
            mode="markers",
            marker=dict(color="Red"),
            text=self.pareto_df["pretty_formula"].to_numpy()[self.ps],
            name="Pareto Optimal",
        )
        scat_2 = Scatter(
            x=x_plot[self.nps, 0],
            y=x_plot[self.nps, 1],
            mode="markers",
            marker=dict(color="Blue"),
            text=self.pareto_df["pretty_formula"].to_numpy()[self.nps],
            name="Non Pareto Optimal",
        )
        self.fig_pareto.update_xaxes(
            title="PCA Reduced dimension 1 with explained variance ratio {}".format(
                pca.explained_variance_ratio_[0]
            )
        )
        self.fig_pareto.update_yaxes(
            title="PCA Reduced dimension 2 with explained variance ratio {}".format(
                pca.explained_variance_ratio_[1]
            )
        )
        self.fig_pareto.add_traces([scat, scat_2])
        with self.output_plot_2:
            self.output_plot_2.clear_output()
            display(self.fig_pareto)

        self.pareto_df.drop(self.nps, axis="rows", inplace=True)
        self.pareto_df.reset_index(inplace=True, drop=True)
        self.pareto_sampling_progress.value = 0
        self.pareto_ct_label = Label(
            value="Number of Pareto Optimal Solutions is {}".format(len(self.ps)),
            layout=self.fs_layout,
        )
        self.plot_top_cand_button.disabled = False
        self.display_all()

    def compute_objectives(self, button_instance):
        def obj_fun(wts: NDArrayFloat, alpha: float, x_vals: NDArrayFloat):
            n_pts, n_met = x_vals.shape
            Z = np.zeros(n_pts)
            Z_purewts = np.zeros(n_pts)
            for i_pt in range(0, n_pts):
                Z[i_pt] = np.dot(x_vals[i_pt], wts) - alpha * np.std(
                    np.multiply(x_vals[i_pt], wts)
                )
                Z_purewts[i_pt] = np.dot(x_vals[i_pt], wts)
            return Z, Z_purewts

        self.output_plot: Output = Output()
        metrics: List[str] = self.pareto_metrics.copy()

        self.metric_weights = []
        for slider in self.weight_sliders:
            self.metric_weights.append(slider.value)

        weights = np.array(self.metric_weights)
        self.X_for_pareto = self.pareto_df[metrics].to_numpy()
        self.paretomeans = np.mean(self.X_for_pareto, axis=0)
        self.paretostds = np.std(self.X_for_pareto, axis=0)
        self.X_for_pareto = (self.X_for_pareto - self.paretomeans) / self.paretostds

        objective, pure_weight = obj_fun(wts=weights, alpha=1, x_vals=self.X_for_pareto)
        n_met = weights.shape[0]
        self.norm_err_bars = np.zeros(self.err_bars.shape)
        for i_met in range(0, n_met):
            if self.pareto_metrics[i_met] == "vol_mag":
                self.norm_err_bars[:, i_met] = (
                    self.err_bars[:, i_met] / self.paretostds[i_met]
                )
            elif self.pareto_metrics[i_met] == "Norm DoS":
                self.norm_err_bars[:, i_met] = (
                    self.err_bars[:, i_met] / self.paretostds[i_met]
                )
            elif self.pareto_metrics[i_met] == "vf":
                self.norm_err_bars[:, i_met] = (
                    self.err_bars[:, i_met] / self.paretostds[i_met]
                )
            else:
                self.norm_err_bars[:, i_met] = (
                    self.err_bars[:, i_met] / self.paretostds[i_met]
                )

        self.pareto_df["objective_fn"] = Series(objective)
        self.pareto_df["unreg_objective_fn"] = Series(pure_weight)

        self.add_to_plot.disabled = False
        self.display_top_five()
        self.pareto_df.sort_index(ascending=True, inplace=True)

        self.display_all()

    def display_top_five(self):
        fig: Figure = go.Figure()
        self.X_for_plot = self.pareto_df[self.selected_metrics].to_numpy()
        self.plotmeans = np.mean(self.X_for_plot, axis=0)
        self.plotstds = np.std(self.X_for_plot, axis=0)
        self.X_for_plot = (self.X_for_plot - self.plotmeans) / self.plotstds

        self.pareto_df.sort_values(by="objective_fn", ascending=False, inplace=True)
        metrics: List[str] = self.selected_metrics.copy()
        traces: List[Scatterpolar] = []

        n_metrics = len(metrics)
        num_top_c = min(10, self.pareto_df.shape[0])
        unorm_values = self.pareto_df[metrics].to_numpy()[0:num_top_c]
        values = (unorm_values - self.plotmeans) / self.plotstds
        formulae = self.pareto_df["pretty_formula"].to_numpy()[0:num_top_c]

        labels = metrics.copy()
        labels.append(labels[0])
        for i_cd in range(0, num_top_c):
            h_info = []
            for i_metric in range(0, n_metrics):
                h_info.append(
                    "{} : {:.3f}".format(
                        metrics[i_metric], unorm_values[i_cd, i_metric]
                    )
                )
            vals = list(values[i_cd])
            vals.append(vals[0])
            h_info.append(h_info[0])

            fig.add_trace(
                go.Scatterpolar(
                    r=vals, theta=labels, name=formulae[i_cd], opacity=0.7, text=h_info
                )
            )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=False),
            ),
            width=800,
            height=600,
            font=dict(size=14),
        )

        self.fig_widget = FigureWidget(fig)
        with self.output_plot:
            self.output_plot.clear_output()
            display(self.fig_widget)

    def add_benchmarks(self, button_instance):
        n_bm = len(self.benchmark_mat)
        df_add = self.df_benchmark.copy()
        for i_bm in range(0, n_bm):
            chbox = self.checkbox_add_to_plot[i_bm]
            if chbox.value == 0:
                df_add.drop(i_bm, axis="rows", inplace=True)
        df_add.reset_index(inplace=True, drop=True)

        metrics: List[str] = self.selected_metrics.copy()
        traces: List[Scatterpolar] = []

        n_metrics = len(metrics)
        num_top_c = df_add.shape[0]
        unorm_values = df_add[self.selected_metrics].to_numpy()
        values = (unorm_values - self.plotmeans) / self.plotstds
        formulae = df_add["pretty_formula"].to_numpy()[0:num_top_c]

        labels = metrics.copy()
        labels.append(labels[0])
        for i_cd in range(0, num_top_c):
            h_info = []
            for i_metric in range(0, n_metrics):
                h_info.append(
                    "{} : {:.3f}".format(
                        metrics[i_metric], unorm_values[i_cd, i_metric]
                    )
                )
            vals = list(values[i_cd])
            vals.append(vals[0])
            h_info.append(h_info[0])

            self.fig_widget.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=labels,
                    name=formulae[i_cd],
                    opacity=0.7,
                    text=h_info,
                    line=dict(dash="dashdot"),
                )
            )

        with self.output_plot:
            self.output_plot.clear_output()
            display(self.fig_widget)

    def display_all(self):
        grid_children = []
        title_garea_string = "title"
        grid_children.append(self.title)
        self.title.layout = Layout(
            height="auto", width="auto", grid_area=title_garea_string
        )
        grid_template_str = '"{}"\n'.format(title_garea_string)

        tmp_hboxes: List[HBox] = []
        n_chboxmetric_rows = int(max(np.floor(len(self.checkbox_metrics) / 3), 0)) + 1
        for row in range(0, n_chboxmetric_rows):
            tmp_hbox: HBox = HBox(
                self.checkbox_metrics[
                    row * 3 : min(row * 3 + 3, len(self.checkbox_metrics))
                ],
                layout=Layout(display="flex", flex_flow="row", align_items="stretch"),
            )
            tmp_hboxes.append(tmp_hbox)

        hbox_cuttofs_box: VBox = VBox(tmp_hboxes)

        tmp_hboxes: List[HBox] = []
        n_chboxpareto_rows = int(max(np.floor(len(self.checkbox_pareto) / 3), 0)) + 1
        if len(self.checkbox_pareto) != 0:
            for row in range(0, n_chboxpareto_rows):
                tmp_hbox: HBox = HBox(
                    self.checkbox_pareto[
                        row * 3 : min(row * 3 + 3, len(self.checkbox_pareto))
                    ],
                    layout=Layout(
                        display="flex", flex_flow="row", align_items="stretch"
                    ),
                )
                tmp_hboxes.append(tmp_hbox)

        hbox_pareto_box: VBox = VBox(tmp_hboxes)

        vbox_cutoff_sliders: HBox = HBox(
            [
                VBox(self.cutoff_sliders, layout=Layout(width="50%")),
                VBox(self.filtered_count_Label, layout=Layout(width="50%")),
            ],
            layout=Layout(
                display="flex",
                flex_flow="row",
                align_items="stretch",
                justify_items="center",
            ),
        )

        vbox_weight_sliders: VBox = VBox(self.weight_sliders)

        output_garea_string = "output"
        grid_children.append(self.output_plot)
        self.output_plot.layout = Layout(
            height="auto", width="auto", grid_area=output_garea_string
        )

        hbox_cutoff_str = "cutoff_metrics"
        grid_children.append(hbox_cuttofs_box)
        hbox_cuttofs_box.layout = Layout(
            height="auto", width="auto", grid_area=hbox_cutoff_str
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(hbox_cutoff_str) * (
            max(len(self.checkbox_metrics) // 3, 0) + 1
        )

        selectmetrics_garea_string = "selectmetrics_button"
        grid_children.append(self.select_metrics_button)
        self.select_metrics_button.layout = Layout(
            height="auto", width="auto", grid_area=selectmetrics_garea_string
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(
            selectmetrics_garea_string
        )

        vbox_cutoff_str = "cutoff_sliders"
        grid_children.append(vbox_cutoff_sliders)
        vbox_cutoff_sliders.layout = Layout(
            height="auto", width="auto", grid_area=vbox_cutoff_str
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(vbox_cutoff_str) * len(
            self.selected_metrics
        )

        selectcutoffs_garea_string = "selectcutoffs_button"
        grid_children.append(self.filter_cutoffs_button)
        self.filter_cutoffs_button.layout = Layout(
            height="auto", width="auto", grid_area=selectcutoffs_garea_string
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(
            selectcutoffs_garea_string
        )

        numpareto_gareastring = "np_lanel"
        if self.pareto_anal_ct_label is not None:
            grid_children.append(self.pareto_anal_ct_label)
            self.pareto_anal_ct_label.layout = Layout(
                height="auto", width="auto", grid_area=numpareto_gareastring
            )
            grid_template_str = grid_template_str + '"{}"\n'.format(
                numpareto_gareastring
            )

        hbox_pareto_str = "pareto_metrics"
        grid_children.append(hbox_pareto_box)
        hbox_pareto_box.layout = Layout(
            height="auto", width="auto", grid_area=hbox_pareto_str
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(hbox_pareto_str) * (
            max(len(self.checkbox_pareto) // 3, 0) + 1
        )

        findpareto_garea_string = "findpareto_button"
        grid_children.append(self.find_paerto_button)
        self.find_paerto_button.layout = Layout(
            height="auto", width="auto", grid_area=findpareto_garea_string
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(findpareto_garea_string)

        if self.pareto_sampling_progress.min != -1:
            paretoprog_garea_string = "pareto_progbar"
            grid_children.append(self.pareto_sampling_progress)
            self.pareto_sampling_progress.layout = Layout(
                height="auto", width="auto", grid_area=paretoprog_garea_string
            )
            grid_template_str = grid_template_str + '"{}"\n'.format(
                paretoprog_garea_string
            )

        numpareto_gareastring = "pareto_num_label"
        if self.pareto_ct_label is not None:
            grid_children.append(self.pareto_ct_label)
            self.pareto_ct_label.layout = Layout(
                height="auto", width="auto", grid_area=numpareto_gareastring
            )
            grid_template_str = grid_template_str + '"{}"\n'.format(
                numpareto_gareastring
            )

        if self.fig_pareto is not None:
            out_plot_pareto_garea_string = "pareto_plot"
            grid_children.append(self.output_plot_2)
            self.output_plot_2.layout = Layout(
                height="auto", width="auto", grid_area=out_plot_pareto_garea_string
            )
            grid_template_str = (
                grid_template_str + '"{}"\n'.format(out_plot_pareto_garea_string) * 10
            )

        vbox_weights_str = "weight_sliders"
        grid_children.append(vbox_weight_sliders)
        vbox_weight_sliders.layout = Layout(
            height="auto", width="auto", grid_area=vbox_weights_str
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(vbox_weights_str) * len(
            self.weight_sliders
        )

        alpha_garea_string = "alpha"
        grid_children.append(self.alpha_slider)
        self.alpha_slider.layout = Layout(
            height="auto", width="auto", grid_area=alpha_garea_string
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(alpha_garea_string)

        computebutton_garea_string = "compute_button"
        grid_children.append(self.plot_top_cand_button)
        self.plot_top_cand_button.layout = Layout(
            height="auto", width="auto", grid_area=computebutton_garea_string
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(
            computebutton_garea_string
        )

        grid_template_str = (
            grid_template_str + '"{}"\n'.format(output_garea_string) * 15
        )

        hbox_bm: HBox = HBox(self.checkbox_add_to_plot)
        hbox_bm_str = "bm_materials"
        grid_children.append(hbox_bm)
        hbox_bm.layout = Layout(height="auto", width="auto", grid_area=hbox_bm_str)
        grid_template_str = grid_template_str + '"{}"\n'.format(hbox_bm_str)

        addplotbutton_garea_string = "addplot_button"
        grid_children.append(self.add_to_plot)
        self.add_to_plot.layout = Layout(
            height="auto", width="auto", grid_area=addplotbutton_garea_string
        )
        grid_template_str = grid_template_str + '"{}"\n'.format(
            addplotbutton_garea_string
        )

        grid_rows = "auto " * (len(grid_children) - 2) + "auto"
        grid_columns = "100%"
        grid = GridBox(
            children=grid_children,
            layout=Layout(
                width="100%",
                height="100%",
                grid_template_columns=grid_columns,
                grid_template_rows=grid_rows,
                # grid_gap='0px',
                grid_template_areas=grid_template_str
                # justify_items='center',
            ),
        )

        with self.gui_out:
            self.gui_out.clear_output()
            display(grid)
