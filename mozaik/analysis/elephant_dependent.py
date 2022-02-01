import elephant
import warnings
import numpy as np
import quantities as pq
from parameters import ParameterSet
from scipy.optimize import curve_fit
from mozaik.storage import queries
from mozaik.analysis.analysis import Analysis
from mozaik.analysis.data_structures import SingleValue, SingleValueList
from mozaik.tools.distribution_parametrization import load_parameters


class CriticalityAnalysis(Analysis):
    """
    Compute distance to criticality as defined in:
    Ma Z., Turrigiano G.G., Wessel R., Hengen K.B. (2019). Cortical Circuit
    Dynamics Are Homeostatically Tuned to Criticality In Vivo, Neuron
    Computes the histogram of population activity per layer and calculates
    the distance to criticality for several intervals.
    Can only be run on stimuli with the same name.
    """

    required_parameters = ParameterSet(
        {
            "num_bins": int,  # Number of duration and size bins
        }
    )

    def perform_analysis(self):
        layers = [["V1_Exc_L2/3", "V1_Inh_L2/3"], ["V1_Exc_L4", "V1_Inh_L4"]]
        for layer in layers:
            if not set(layer).issubset(set(self.datastore.sheets())):
                warnings.warn(
                    "Layer %s not part of data store sheets: %s!"
                    % (layer, self.datastore.sheets())
                )
                continue

            # pool all spikes from the layer
            allspikes = []
            dsv = queries.param_filter_query(self.datastore, sheet_name=layer)
            segs = dsv.get_segments()
            # add spikes from the layer to the pool
            tstart = tstop = 0
            for seg in segs:
                for st in seg.spiketrains:
                    tstart = min(st.t_start.magnitude,tstart)
                    tstop = max(st.t_stop.magnitude,tstop)
                    allspikes.extend(st.magnitude)

            assert (
                len({load_parameters(str(s))["name"] for s in dsv.get_stimuli()}) == 1
            ), "All stimuli have to have the same name!"
            # calculate specific time bin in each segment as in Fontenele2019
            dt = np.mean(np.diff(np.sort(allspikes)))

            # case with no spikes is not taken care off!
            bins = np.arange(tstart, tstop, dt)
            hist, bins = np.histogram(allspikes, bins)

            # find zeros in the histogram
            zeros = np.where(hist == 0)[0]

            # calculate durations of avalanches
            durs = np.diff(zeros) * dt
            durs = durs[durs > dt]

            # calculate sizes of avalanches
            szs = []
            for i in range(len(zeros) - 1):
                szs.append(np.sum(hist[zeros[i] : zeros[i + 1]]))  # .magnitude))
            szs = np.array(szs)
            szs = szs[szs > 0]

            # calculate tau=exponent of size distr
            s_distr, s_bins = self.create_hist(szs, self.parameters.num_bins)
            s_amp, s_slope, s_error_sq, s_error_diff = self.fit_powerlaw_distribution(
                s_bins, s_distr, "size"
            )

            # calculate tau_t=exponent of distr of durations
            d_distr, d_bins = self.create_hist(durs, self.parameters.num_bins)
            d_amp, d_slope, d_error_sq, d_error_diff = self.fit_powerlaw_distribution(
                d_bins, d_distr, "duration"
            )

            # calculate the <S>(D) curve, S=size, D=duration
            [sd_amp, sd_slope], _ = curve_fit(
                f=self.powerlaw, xdata=durs, ydata=szs, p0=[0, 0]
            )

            beta = sd_slope  # for dcc calculation
            error_diff = sum(szs - self.powerlaw(durs, sd_amp, sd_slope))
            error_sq = np.linalg.norm(szs - self.powerlaw(durs, sd_amp, sd_slope))
            crit_dist = np.abs(beta - (-d_slope - 1) / (-s_slope - 1))

            stims = dsv.get_stimuli()
            common_stim_params = load_parameters(str(stims[0]))
            for st in stims:
                p = load_parameters(str(st))
                common_stim_params = {  # Inner join of 2 dicts
                    k: v
                    for (k, v) in p.items()
                    if k in common_stim_params and p[k] == common_stim_params[k]
                }

            for sheet in layer:
                common_params = {
                    "sheet_name": sheet,
                    "tags": self.tags,
                    "analysis_algorithm": self.__class__.__name__,
                    "stimulus_id": str(common_stim_params),
                }
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=crit_dist * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="DistanceToCriticality",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=dt * pq.s,
                        value_units=pq.s,
                        value_name="AvalancheBinSize",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValueList(
                        values=durs * pq.s,
                        values_unit=pq.s,
                        value_name="AvalancheDurations",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValueList(
                        values=szs * pq.dimensionless,
                        values_unit=pq.dimensionless,
                        value_name="AvalancheSizes",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=sd_amp * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SDAmplitude",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=sd_slope * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SDSlope",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=error_sq * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SDErrorSq",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=error_diff * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SDErrorDiff",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=s_slope * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SSlope",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=s_amp * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SAmplitude",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValueList(
                        values=s_distr * pq.dimensionless,
                        values_unit=pq.dimensionless,
                        value_name="SDistr",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValueList(
                        values=s_bins * pq.dimensionless,
                        values_unit=pq.dimensionless,
                        value_name="SBins",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=s_error_sq * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SErrorSq",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=s_error_diff * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="SErrorDiff",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=d_slope * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="DSlope",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=d_amp * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="DAmplitude",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValueList(
                        values=d_distr * pq.s,
                        values_unit=pq.dimensionless,
                        value_name="DDistr",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValueList(
                        values=d_bins * pq.s,
                        values_unit=pq.dimensionless,
                        value_name="DBins",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=d_error_sq * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="DErrorSq",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=d_error_diff * pq.dimensionless,
                        value_units=pq.dimensionless,
                        value_name="DErrorDiff",
                        **common_params,
                    )
                )

    def create_hist(self, data, nrbins):
        distr, b = np.histogram(data, bins=nrbins, density=True)
        bs = b[1] - b[0]
        bins = b[:-1] + bs / 2.0
        return distr, bins

    def fit_powerlaw_distribution(self, x, y, img_title=None):
        """
        Parameters
        ----------
        data : 1D numpy array
            Observations from the probability distribution we want to fit
        nrbins : int
            Number of bins in the created histogram
        img_title : str
            Used for debugging. Title of the powerlaw fit figure.

        Returns
        -------
        amp, slope, tau, error

        amp : float
            Amplitude of the powerlaw distribution
        slope : float
            Slope of the powerlaw distribution
        tau : float
            tau = -slope
        error : float
            Mean square error of the fit
        """
        try:
            [amp, slope], _ = curve_fit(f=self.powerlaw, xdata=x, ydata=y, p0=[0, 0])
            if np.isnan(amp) or np.isnan(slope):
                raise RuntimeError("scipy.curve_fit returned nan")
            error_sq = np.linalg.norm(y - self.powerlaw(x, amp, slope))
            error_diff = sum(y - self.powerlaw(x, amp, slope))
            return amp, slope, error_sq, error_diff
        except Exception as e:
            warnings.warn(
                "While fitting the powerlaw distribution, the following exception occured: %s"
                % e
            )
            return 0, 0, 0, 0

    @staticmethod
    def powerlaw(x, amp, slope):
        return amp * np.power(x, slope)

    def debug_plot(self, binss, distr, amp, slope, title):
        import pylab

        fig = pylab.figure()
        ax = pylab.gca()
        ax.plot(binss, self.powerlaw(binss, amp, slope))
        ax.plot(binss, distr, "o")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xscale("log")
        pylab.savefig("%s.png" % title)
        pylab.close()
