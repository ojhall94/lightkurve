"""Defines the Periodogram class and associated tools."""
from __future__ import division, print_function

import copy
import logging

import numpy as np
from matplotlib import pyplot as plt

import astropy
from astropy.table import Table
from astropy.stats import LombScargle
from astropy import __version__
from astropy import units as u
from astropy.units import cds
from astropy.convolution import convolve, Box1DKernel

from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

from .asteroseismology import dnu_mass_prior, estimate_mass, estimate_radius, estimate_logg

from . import MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['Periodogram']


class Periodogram(object):
    """Class to represents a power spectrum, i.e. frequency vs power.

    The Periodogram class represents a power spectrum, with values of
    frequency on the x-axis (in any frequency units) and values of power on the
    y-axis (in units of ppm^2 / [frequency units]).

    Attributes
    ----------
    frequency : `astropy.units.Quantity` object
        Array of frequencies with associated astropy unit.
    power : `astropy.units.Quantity` object
        Array of power-spectral-densities. The Quantity array must have units
        of `ppm^2 / freq_unit`, where freq_unit is the unit of the frequency
        attribute.
    nyquist : float, optional
        The Nyquist frequency of the lightcurve. In units of freq_unit, where
        freq_unit is the unit of the frequency attribute.
    targetid : str, optional
        Identifier of the target.
    label : str, optional
        Human-friendly object label, e.g. "KIC 123456789".
    meta : dict, optional
        Free-form metadata associated with the Periodogram.
    """
    def __init__(self, frequency, power, nyquist=None, label=None,
                 targetid=None, meta={}):
        # Input validation
        if not isinstance(frequency, u.quantity.Quantity):
            raise ValueError('frequency must be an `astropy.units.Quantity` object.')
        if not isinstance(power, u.quantity.Quantity):
            raise ValueError('power must be an `astropy.units.Quantity` object.')
        # Frequency must have frequency units
        try:
            frequency.to(u.Hz)
        except u.UnitConversionError:
            raise ValueError('Frequency must be in units of 1/time.')
        # Frequency and power must have sensible shapes
        if frequency.shape[0] <= 1:
            raise ValueError('frequency and power must have a length greater than 1.')
        if frequency.shape != power.shape:
            raise ValueError('frequency and power must have the same length.')

        self.frequency = frequency
        self.power = power
        self.nyquist = nyquist
        self.label = label
        self.targetid = targetid
        self.meta = meta

    @property
    def period(self):
        """Returns the array of periods, i.e. 1/frequency."""
        return 1. / self.frequency

    @property
    def max_power(self):
        """Returns the power of the highest peak in the periodogram."""
        return np.nanmax(self.power)

    @property
    def frequency_at_max_power(self):
        """Returns the frequency corresponding to the highest peak in the periodogram."""
        return self.frequency[np.nanargmax(self.power)]

    @property
    def period_at_max_power(self):
        """Returns the period corresponding to the highest peak in the periodogram."""
        return 1. / self.frequency_at_max_power

    @staticmethod
    def from_lightcurve(lc, min_frequency=None, max_frequency=None,
                        min_period=None, max_period=None,
                        frequency=None, period=None,
                        nterms=1, nyquist_factor=1, oversample_factor=1,
                        freq_unit=1/u.day, **kwargs):
        """Creates a Periodogram from a LightCurve using the Lomb-Scargle method.

        By default, the periodogram will be created for a regular grid of
        frequencies from one frequency separation to the Nyquist frequency,
        where the frequency separation is determined as 1 / the time baseline.

        The min frequency and/or max frequency (or max period and/or min period)
        can be passed to set custom limits for the frequency grid. Alternatively,
        the user can provide a custom regular grid using the `frequency`
        parameter or a custom regular grid of periods using the `period`
        parameter.

        The spectrum can be oversampled by increasing the oversample_factor
        parameter. The parameter nterms controls how many Fourier terms are used
        in the model. Note that many terms could lead to spurious peaks. Setting
        the Nyquist_factor to be greater than 1 will sample the space beyond the
        Nyquist frequency, which may introduce aliasing.

        The unit parameter allows a request for alternative units in frequency
        space. By default frequency is in (1/day) and power in (ppm^2 * day).
        Asteroseismologists for example may want frequency in (microHz) and
        power in (ppm^2 / microHz), in which case they would pass
        `unit = u.microhertz` where `u` is `astropy.units`

        By default this method uses the LombScargle 'fast' method, which assumes
        a regular grid. If a regular grid of periods (i.e. an irregular grid of
        frequencies) it will use the 'slow' method. If nterms > 1 is passed, it
        will use the 'fastchi2' method for regular grids, and 'chi2' for
        irregular grids. The normalizatin of the Lomb Scargle periodogram is
        fixed to `psd`, and cannot be overridden.

        Caution: this method assumes that the LightCurve's time (lc.time)
        is given in units of days.

        Parameters
        ----------
        lc : LightCurve object
            The LightCurve from which to compute the Periodogram.
        min_frequency : float
            If specified, use this minimum frequency rather than one over the
            time baseline.
        max_frequency : float
            If specified, use this maximum frequency rather than nyquist_factor
            times the nyquist frequency.
        min_period : float
            If specified, use 1./minium_period as the maximum frequency rather
            than nyquist_factor times the nyquist frequency.
        max_period : float
            If specified, use 1./maximum_period as the minimum frequency rather
            than one over the time baseline.
        frequency :  array-like
            The regular grid of frequencies to use. If given a unit, it is
            converted to units of freq_unit. If not, it is assumed to be in
            units of freq_unit. This over rides any set frequency limits.
        period : array-like
            The regular grid of periods to use (as 1/period). If given a unit,
            it is converted to units of freq_unit. If not, it is assumed to be
            in units of 1/freq_unit. This overrides any set period limits.
        nterms : int
            Default 1. Number of terms to use in the Fourier fit.
        nyquist_factor : int
            Default 1. The multiple of the average Nyquist frequency. Is
            overriden by maximum_frequency (or minimum period).
        oversample_factor : int
            The frequency spacing, determined by the time baseline of the
            lightcurve, is divided by this factor, oversampling the frequency
            space. This parameter is identical to the samples_per_peak parameter
            in astropy.LombScargle()
        freq_unit : `astropy.units.core.CompositeUnit`
            Default: 1/u.day. The desired frequency units for the Lomb Scargle
            periodogram. This implies that 1/freq_unit is the units for period.
        kwargs : dict
            Keyword arguments passed to `astropy.stats.LombScargle()`

        Returns
        -------
        Periodogram : `Periodogram` object
            Returns a Periodogram object extracted from the lightcurve.
        """
        # Make sure the lightcurve object is normalized
        lc = lc.normalize()

        # Check if any values of period have been passed and set format accordingly
        if not all(b is None for b in [period, min_period, max_period]):
            format = 'period'
        else:
            format = 'frequency'

        # If period and frequency keywords have both been set, throw an error
        if (not all(b is None for b in [period, min_period, max_period])) & \
           (not all(b is None for b in [frequency, min_frequency, max_frequency])):
            raise ValueError('You have input keyword arguments for both frequency and period. '
                             'Please only use one.')

        if (~np.isfinite(lc.flux)).any():
            raise ValueError('Lightcurve contains NaN values. Use lc.remove_nans()'
                             ' to remove NaN values from a LightCurve.')

        # Hard coding that time is in days.
        time = lc.time.copy() * u.day

        # Calculate Nyquist Frequency and frequency bin width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(time))))
        fs = (1./(time[-1] - time[0])) / oversample_factor

        # Convert these values to requested frequency unit
        nyquist = nyquist.to(freq_unit)
        fs = fs.to(freq_unit)

        # Warn if there is confusing input
        if (frequency is not None) & (any([a is not None for a in [min_frequency, max_frequency]])):
            log.warning("You have passed both a grid of frequencies "
                        "and min_frequency/max_frequency arguments; "
                        "the latter will be ignored.")
        if (period is not None) & (any([a is not None for a in [min_period, max_period]])):
            log.warning("You have passed a grid of periods "
                        "and min_period/max_period arguments; "
                        "the latter will be ignored.")

        # Tidy up the period stuff...
        if max_period is not None:
            # min_frequency MUST be none by this point.
            min_frequency = 1. / max_period
        if min_period is not None:
            # max_frequency MUST be none by this point.
            max_frequency = 1. / min_period
        # If the user specified a period, copy it into the frequency.
        if (period is not None):
            frequency = 1. / period

        # Do unit conversions if user input min/max frequency or period
        if frequency is None:
            if min_frequency is not None:
                min_frequency = u.Quantity(min_frequency, freq_unit)
            if max_frequency is not None:
                max_frequency = u.Quantity(max_frequency, freq_unit)
            if (min_frequency is not None) & (max_frequency is not None):
                if (min_frequency > max_frequency):
                    if format == 'frequency':
                        raise ValueError('min_frequency cannot be larger than max_frequency')
                    if format == 'period':
                        raise ValueError('min_period cannot be larger than max_period')
            # If nothing has been passed in, set them to the defaults
            if min_frequency is None:
                min_frequency = fs
            if max_frequency is None:
                max_frequency = nyquist * nyquist_factor

            # Create frequency grid evenly spaced in frequency
            frequency = np.arange(min_frequency.value, max_frequency.value, fs.to(freq_unit).value)

        # Convert to desired units
        frequency = u.Quantity(frequency, freq_unit)

        if nterms > 1:
            raise NotImplementedError('Increasing the number of terms is not implemented yet.')
        else:
            method = 'fast'

        if period is not None:
            method = 'slow'
            log.warning("You have passed an evenly-spaced grid of periods. "
                        "These are not evenly spaced in frequency space.\n"
                        "Method has been set to 'slow' to allow for this.")

        if float(__version__[0]) >= 3:
            LS = LombScargle(time, lc.flux * 1e6,
                             nterms=nterms, normalization='psd', **kwargs)
            power = LS.power(frequency, method=method)
        else:
            LS = LombScargle(time, lc.flux * 1e6,
                             nterms=nterms, **kwargs)
            power = LS.power(frequency, method=method, normalization='psd')

        # Normalise the according to Parseval's theorem
        norm = np.std(lc.flux * 1e6)**2 / np.sum(power)
        power *= norm

        power = power * (cds.ppm**2)

        # Rescale power to units of ppm^2 / [frequency unit]
        power = power / fs

        # Periodogram needs properties
        return Periodogram(frequency=frequency, power=power, nyquist=nyquist,
                           targetid=lc.targetid, label=lc.label)

    def bin(self, binsize=10, method='mean'):
        """Bins the power spectrum.

        Parameters
        ----------
        binsize : int
            The factor by which to bin the power spectrum, in the sense that
            the power spectrum will be smoothed by taking the mean in bins
            of size N / binsize, where N is the length of the original
            frequency array. Defaults to 10.
        method : str, one of 'mean' or 'median'
            Method to use for binning. Default is 'mean'.

        Returns
        -------
        binned_periodogram : a `Periodogram` object
            Returns a new `Periodogram` object which has been binned.
        """
        # Input validation
        if binsize < 1:
            raise ValueError('binsize must be larger than or equal to 1')
        if method not in ('mean', 'median'):
            raise ValueError("{} is not a valid method, must be 'mean' or 'median'.".format(method))

        m = int(len(self.power) / binsize)  # length of the binned arrays
        if method == 'mean':
            binned_freq = self.frequency[:m*binsize].reshape((m, binsize)).mean(1)
            binned_power = self.power[:m*binsize].reshape((m, binsize)).mean(1)
        elif method == 'median':
            binned_freq = np.nanmedian(self.frequency[:m*binsize].reshape((m, binsize)), axis=1)
            binned_power = np.nanmedian(self.power[:m*binsize].reshape((m, binsize)), axis=1)

        binned_pg = copy.deepcopy(self)
        binned_pg.frequency = binned_freq
        binned_pg.power = binned_power
        return binned_pg

    def smooth(self, filter_width = 0.1/u.day):
        """Smooths the power spectrum by convolving with a numpy Box1DKernel.
        This method requires a Periodogram class built using an evenly
        spaced grid of periods.

        Parameters
        ----------
        filter_width : float
            The width of the smoothing filter in units of frequency. If given
            astropy units of frequency, will convert to the same units as the
            Periodogram frequency attribute. If unitless, will be given same
            units as the Periodogram frequency attribute.

        Returns
        -------
        smoothed_pg : a `Periodogram` object
            Returns a new `Periodogram` object which has been smoothed.
        """
        # Input validation
        if filter_width <= 0.:
            raise ValueError('Filter width must be larger than 0')

        try:
            filter_width = u.Quantity(filter_width, self.frequency.unit)
        except u.UnitConversionError:
            raise ValueError('filter_width must be in units of frequency.')

        fs = np.mean(np.diff(self.frequency))

        #Check to see if we have a grid of evenly spaced periods instead.
        if np.isclose(np.median(np.diff(self.frequency.value)), fs.value):
            box_kernel = Box1DKernel(np.ceil(filter_width/fs))
            smooth_power = convolve(self.power.value, box_kernel)
            smooth_power = u.Quantity(smooth_power, self.power.unit)

            smooth_pg = copy.deepcopy(self)
            smooth_pg.power = smooth_power
        else:
            raise NotImplementedError("The smooth() function requires a grid of evenly spaced frequencies at this time.")
        return smooth_pg

    def plot(self, scale='linear', ax=None, xlabel=None, ylabel=None, title='',
             style='lightkurve', format='frequency', unit=None, **kwargs):
        """Plots the Periodogram.

        Parameters
        ----------
        scale: str
            Set x,y axis to be "linear" or "log". Default is linear.
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        format : str
            {'frequency', 'period'}. Default 'frequency'. If 'frequency', x-axis
            units will be frequency. If 'period', the x-axis units will be
            period and 'log' scale.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if isinstance(unit, u.quantity.Quantity):
            unit = unit.unit

        if unit is None:
            unit = self.frequency.unit
            if format == 'period':
                unit = self.period.unit

        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        if ylabel is None:
            ylabel = "Power Spectral Density [{}]".format(self.power.unit.to_string('latex'))

        # This will need to be fixed with housekeeping. Self.label currently doesnt exist.
        if ('label' not in kwargs) and ('label' in dir(self)):
            kwargs['label'] = self.label

        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot frequency and power
            if format.lower() == 'frequency':
                ax.plot(self.frequency.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Frequency [{}]".format(unit.to_string('latex'))
            elif format.lower() == 'period':
                ax.plot(self.period.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Period [{}]".format(unit.to_string('latex'))
            else:
                raise ValueError('{} is not a valid plotting format'.format(format))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if (np.sum([len(a) for a in legend_labels]) != 0):
                ax.legend()
            ax.set_yscale(scale)
            ax.set_xscale(scale)
            ax.set_title(title)
        return ax

    def _estimate_background(self, log_width=0.01):
        """Estimates the background noise of the power spectrum.

        This method uses a moving filter in log10 space. The filter defines
        a bin centered at a value x0 with a spread of ``log_width`` either side.
        The median of the power in this bin will be added to all indices
        within the bin in an empty array, `bkg`.
        The bin then moves along in a step of x0 + 0.5 * log_width. This means
        that each index will contain the sum of multiple medians of bins that
        index is included in. To normalize this, we divide the background value
        in each index by the number of median values that were added to that
        index.

        Parameters
        ----------
        log_width : float
            Default 0.01. The width of the filter in log10 space.

        Returns
        -------
        bkg : array-like
            An estimate of the noise background of the power spectrum. Has the
            same units as the `power` attribute.
        """
        if isinstance(self.frequency, astropy.units.quantity.Quantity):
            f = self.frequency.value
        else:
            f = self.frequency
        if isinstance(self.power, astropy.units.quantity.Quantity):
            p = self.power.value
        else:
            p = self.power

        count = np.zeros(len(f), dtype=int)
        bkg = np.zeros_like(f)
        x0 = np.log10(f[0])
        while x0 < np.log10(f[-1]):
            m = np.abs(np.log10(f) - x0) < log_width
            if len(bkg[m] > 0):
                bkg[m] += np.nanmedian(p[m])
                count[m] += 1
            x0 += 0.5 * log_width
        return bkg / count

    def estimate_snr(self, log_width=0.01, return_trend=False):
        """Estimates the Signal-To-Noise (SNR) spectrum.

        This method divides the power spectrum by a background estimated
        using a moving filter in log10 space.

        Parameters
        ----------
        log_width : float
            Default 0.01. The width of the filter in log10 space. Kwarg for the
            Periodogram.estimate_background() function.

        Returns
        -------
        snr_spectrum : a `Periodogram` object
            Returns a periodogram object where the power is an estimate of the
            signal-to-noise of the spectrum, assuming a simple estimate of the
            noise background using a moving filter in log10 space.
        """
        bkg = u.Quantity(self._estimate_background(log_width=log_width), self.power.unit)
        snr_pg = self / bkg
        snr = SNRPeriodogram(snr_pg.frequency, snr_pg.power,
                             nyquist=self.nyquist, targetid=self.targetid,
                             label=self.label, meta=self.meta)
        if return_trend:
            bkg = Periodogram(snr_pg.frequency, bkg,
                              nyquist=self.nyquist, targetid=self.targetid,
                              label=self.label, meta=self.meta)
            return snr, bkg
        return snr

    def to_table(self):
        """Exports the Periodogram as an Astropy Table.

        Returns
        -------
        table : `astropy.table.Table` object
            An AstroPy Table with columns 'frequency', 'period', and 'power'.
        """
        return Table(data=(self.frequency, self.period, self.power),
                     names=('frequency', 'period', 'power'),
                     meta=self.meta)

    def __repr__(self):
        return('Periodogram(ID: {})'.format(self.targetid))

    def __getitem__(self, key):
        copy_self = copy.copy(self)
        copy_self.frequency = self.frequency[key]
        copy_self.power = self.power[key]
        return copy_self

    def __add__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = copy_self.power + u.Quantity(other, self.power.unit)
        return copy_self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = other - copy_self.power
        return copy_self

    def __mul__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = other * copy_self.power
        return copy_self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1./other)

    def __rtruediv__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = other / copy_self.power
        return copy_self

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def properties(self):
        """Prints a summary of the non-callable attributes of the Periodogram object.

        Prints in order of type (ints, strings, lists, arrays and others).
        Prints in alphabetical order.
        """
        attrs = {}
        for attr in dir(self):
            if not attr.startswith('_'):
                res = getattr(self, attr)
                if callable(res):
                    continue

                if isinstance(res, astropy.units.quantity.Quantity):
                    unit = res.unit
                    res = res.value
                    attrs[attr] = {'res': res}
                    attrs[attr]['unit'] = unit.to_string()
                else:
                    attrs[attr] = {'res': res}
                    attrs[attr]['unit'] = ''

                if attr == 'hdu':
                    attrs[attr] = {'res': res, 'type': 'list'}
                    for idx, r in enumerate(res):
                        if idx == 0:
                            attrs[attr]['print'] = '{}'.format(r.header['EXTNAME'])
                        else:
                            attrs[attr]['print'] = '{}, {}'.format(
                                attrs[attr]['print'], '{}'.format(r.header['EXTNAME']))
                    continue

                if isinstance(res, int):
                    attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'int'
                elif isinstance(res, float):
                    attrs[attr]['print'] = '{}'.format(np.round(res, 4))
                    attrs[attr]['type'] = 'float'
                elif isinstance(res, np.ndarray):
                    attrs[attr]['print'] = 'array {}'.format(res.shape)
                    attrs[attr]['type'] = 'array'
                elif isinstance(res, list):
                    attrs[attr]['print'] = 'list length {}'.format(len(res))
                    attrs[attr]['type'] = 'list'
                elif isinstance(res, str):
                    if res == '':
                        attrs[attr]['print'] = '{}'.format('None')
                    else:
                        attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'str'
                elif attr == 'wcs':
                    attrs[attr]['print'] = 'astropy.wcs.wcs.WCS'.format(attr)
                    attrs[attr]['type'] = 'other'
                else:
                    attrs[attr]['print'] = '{}'.format(type(res))
                    attrs[attr]['type'] = 'other'

        output = Table(names=['Attribute', 'Description', 'Units'], dtype=[object, object, object])
        idx = 0
        types = ['int', 'str', 'float', 'list', 'array', 'other']
        for typ in types:
            for attr, dic in attrs.items():
                if dic['type'] == typ:
                    output.add_row([attr, dic['print'], dic['unit']])
                    idx += 1
        print('lightkurve.Periodogram properties:')
        output.pprint(max_lines=-1, max_width=-1)


class SNRPeriodogram(Periodogram):
    """Defines a Signal-to-Noise Ratio (SNR) Periodogram class.

    This class is nearly identical to the standard :class:`Periodogram` class,
    but has different plotting defaults.
    """
    def __init__(self, *args, **kwargs):
        super(SNRPeriodogram, self).__init__(*args, **kwargs)

    def __repr__(self):
        return('SNRPeriodogram(ID: {})'.format(self.targetid))

    def plot(self, **kwargs):
        """Plot the SNR spectrum using matplotlib's `plot` method.
        See `Periodogram.plot` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments ot be passed to `Periodogram.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        ax = super(SNRPeriodogram, self).plot(**kwargs)
        if 'ylabel' not in kwargs:
            ax.set_ylabel("Signal to Noise Ratio (SNR)")
        return ax

    def estimate_numax(self, method = 'simple'):
        """Estimates the peak of the envelope of seismic oscillation modes,
        numax, using a choice of method.

        Method `simple` smoothes the periodogram power using a broad Gaussian
        filter. The power excess around the mode envelope creates a hump in the
        smoothed spectrum, the peak of which is taken as numax.

        Method `autocorrelate` first creates an array of possible numax values.
        It then estimates the width of the mode envelope at each numax using a
        standard relation, and autocorrelates this region. For a numax around
        the true numax, this will provide the correlation of the mode envelope
        with itself. Due to the equally spaced pattern of modes, the correlation
        at this numax will be large, revealing the likely numax value.

        Parameters:
        -----------
        method : str
            {'simple', 'autocorrelation'}. Default: 'simple'.

        Returns:
        --------
        numax : float
            The numax of the periodogram. In the units of the periodogram object
            frequency.
        numax_err : float
            The uncertainty on the numax estimate.
        """
        if method == 'simple':
            numax, numax_err, _, _, _ = self._numax_simple()
            return numax, numax_err

        if method == 'autocorrelate':
            raise NotImplementedError('Autocorrelation numax calculation not yet impelmented')

    def _numax_simple(self):
        """Smoothes the periodogram using a broad Gaussian filter, and returns
        the frequency of highest power in the smoothed spectrum"""
        smoothed_ps = gaussian_filter(self.power.value, 1000)
        best_numax = self.frequency[np.argmax(smoothed_ps)]

        #Fit a simple gaussian to the peak
        fwhm = 0.66 * best_numax.value**0.88                    #Predicted FWHM of the envelope
        #Relationship between FWHM and standard deviation. We use half the FWHM to generate sigma.
        sigma_guess = 0.5 * fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
        sel = np.where((self.frequency.value > (best_numax.value-fwhm))
                        * ( self.frequency.value < (best_numax.value+fwhm)))
        popt, pcov = curve_fit(self._gaussian, self.frequency[sel], smoothed_ps[sel],
                                p0 = [sigma_guess, np.max(smoothed_ps), best_numax.value])

        numax = u.Quantity(popt[2], self.frequency.unit)
        numax_err = u.Quantity(popt[0], self.frequency.unit)

        return numax, numax_err, smoothed_ps, popt, sel

    def _numax_autocorrelate(self):
        raise NotImplementedError('Not yet implemented')

    def plot_numax_diagnostics(self, method='simple', **kwargs):
        """Estimates the numax of the oscillation modes, and plots a number of
        diagnostics used in the estimation method.

        For full details on the method please see the docstring for the
        `estimate_numax()` function.

        Parameters:
        -----------
        method : str
            {'simple', 'autocorrelation'}. Default: 'simple'.

        **kwargs : dict
            Dictionary of arguments ot be passed to `Periodogram.plot`.

        Returns:
        --------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.

        """
        if method == 'simple':
            numax, numax_err, smoothed_ps, popt, sel = self._numax_simple()

            ax  = self.plot(**kwargs)
            ax.plot(self.frequency, smoothed_ps, label = 'Filtered PS')
            ax.plot(self.frequency[sel], self._gaussian(self.frequency[sel].value, *popt),
                    label = 'Gaussian Fit')
            ax.axvline(numax.value, linestyle='-.',
                    label="$\\nu_\mathrm{{max}} = {0:.2f}${1}".format(numax.value, numax.unit.to_string('latex')))
            ax.legend()

            return ax

        elif method == 'autocorrelate':
            raise NotImplementedError('Not yet implemented')

    def estimate_dnu(self, method='empirical', numax=None):
        """ Estimates the average value of the large frequency spacing, DeltaNu,
        of the seismic oscillations of the target.

        If method = `empirical`, Dnu will be calculated using an empirical
        relation for numax taken from Stello et al. 2009, as

        dnu = 0.294 * numax^0.772,

        with a 15% uncertainty.

        If method = `autocorrelate`, it will autocorrelate the region around
        the estimated numax expected to contain seismic oscillation modes.
        Repeating peaks in the autocorrelation implies an evenly spaced structure
        of modes. The peak closest to an empirical estimate of dnu is taken as
        the true value.

        If `numax` is None, a simple `numax` is calculated using the
        estimate_numax(method='simple') function.

        Parameters:
        ----------
        method : str
            {'empirical', 'autocorrelation'}. Default: 'simple'.

        numax : float
            An estimated numax value of the mode envelope in the periodogram. If
            not given units it is assumed to be in units of the periodogram
            frequency attribute.

        Returns:
        -------
        deltanu : float
            The average large frequency spacing of the seismic oscillation modes.
            In units of the periodogram frequency attribute.
        """
        if numax is None:
            numax, _ = self.estimate_numax(method='simple')

        if method == 'empirical':
            #Calcluate dnu using the method by Stello et al. 2009
            dnu = u.Quantity(0.294 * numax.value ** 0.772, numax.unit)
            dnu_err = 0.15 * dnu
            return dnu, dnu_err

        if method == 'autocorrelate':
            dnu, dnu_err, _, _, _, _, _, _  = self._dnu_autocorrelate(numax)
            return dnu, dnu_err

    def _dnu_autocorrelate(self, numax):
        """Calculates delta nu by correlating the region expected to contain
        seismic modes with itself."""
        #Calculate the ACF for the best numax estimate
        acf = self.autocorrelate(numax)
        #Finding Mass-prior limits on dnu values for the best numax esimate
        lower, upper = dnu_mass_prior(numax.value)

        #Note that this is only functional for evenly spaced grid of frequencies
        #An exception is already built into self.autocorrelate to check for this, however
        fs = np.median(np.diff(self.frequency))

        #Calculating the correpsonding indices
        lo = int(np.floor(lower / fs.value))
        up = int(np.floor(upper / fs.value))

        #Building list of possible dnus
        lags = np.arange(len(acf)) * fs.value
        acfrange = acf[lo:up]     #The acf range to look for dnu
        lagrange = lags[lo:up]    #The range of lags to look for dnu

        #The best dnu value is at the position of maximum acf power within the range
        best_dnu = lagrange[np.argmax(acfrange)]
        sigma_guess = 0.05 * best_dnu
        # best_dnu, sigma_guess = self.estimate_dnu(method='empirical', numax=numax)

        #Fit a Gaussian to the peak
        sel = np.where((lagrange > (best_dnu - sigma_guess))
                        & (lagrange < (best_dnu + sigma_guess)))
        popt, pcov = curve_fit(self._gaussian, lagrange[sel], acfrange[sel],
                                p0 = [sigma_guess, np.max(acfrange), best_dnu])

        dnu = u.Quantity(popt[2], self.frequency.unit)
        dnu_err = u.Quantity(popt[0], self.frequency.unit)
        return dnu, dnu_err, lags, acf, lo, up, popt, sel

    def plot_dnu_diagnostics(self, method='autocorrelate', numax=None):
        if numax is None:
            numax, _ = self.estimate_numax(method='simple')

        if method == 'empirical':
            raise NotImplentedError('No diagnostic plots for `empirical` method.')

        elif method == 'autocorrelate':
            dnu, dnu_err, lags, acf, lo, up, popt, sel = self._dnu_autocorrelate(numax)

            #TODO: Make these fit the style

            fig, ax = plt.subplots(2,figsize=(12,12))
            self.plot(ax=ax[0])

            ax[1].plot(lags, acf)
            ax[1].plot(lags[lo:up][sel], self._gaussian(lags[lo:up][sel], popt[0], popt[1], popt[2]))
            ax[1].set_xlabel(r'$\Delta\nu$')
            ax[1].set_ylabel(r'ACF')
            ax[1].axvline(lo*np.median(np.diff(self.frequency)).value)
            ax[1].axvline(up*np.median(np.diff(self.frequency)).value)

            fig.tight_layout()
            plt.show()
            return ax

    def estimate_stellar_parameters(self, Teff, Teff_err = None,
                                    numax=None, numax_err=None,
                                    dnu=None, dnu_err=None,
                                    fdnu=1., fnumax=1.):
        """Returns stellar parameters calculated using asteroseismology and a
        value of temperature.

        If no numax or delta nu values are passed, it will calculate them using
        the most precise methods implemented.

        Parameters:
        -----------
        Teff : float
            The effective temperature of the star. In units of Kelvin.
        Teff_error : float
            Error on the Teff value,
        numax : float
            The frequency of maximum power of the seismic mode envelope. In units of
            microhertz.
        numax_error : float
            Error on the numax value.
        dnu : float
            The frequency spacing between two consecutive overtones of equal radial
            degree. In units of microhertz.
        dnu_error : float
            Error on the dnu value.
        fdnu : int
            A correction to the seismic scaling relation for Delta Nu. Effectively
            rescales the solar value for Delta Nu.
        fnumax : int
            A correction to the seismic scaling relation for Numax. Effectively
            rescales the solar value for Numax.

        Returns:
        -------
        stellar_paramaters : dict
            A dictionary containing Mass, Radius, Numax, DeltaNu, and log(g)
            (where `g` is the surface gravity of the star), all with errors.
            Also includes input Teff, and uncertainty on Teff.
        """
        #Make checks
        if (numax is None) & (numax_err is not None):
            raise ValueError('You cant pass a numax error without a numax value!')
        if (dnu is None) & (dnu_err is not None):
            raise ValueError('You cant pass a dnu error without a dnu value!')
        if fdnu <= 0.:
            raise ValueError('fdnu must be larger than 0.')
        if fnumax <= 0.:
            raise ValueError('fnumax must be larger than 0.')

        if numax is None:
            numax, numax_err = self.estimate_numax(method='simple')
        if dnu is None:
            dnu, dnu_err = self.estimate_dnu(method='autocorrelate')

        mass, mass_err =  estimate_mass(numax, dnu, Teff,
                    numax_err=numax_err, dnu_err = dnu_err, Teff_err=Teff_err,
                    fdnu=fdnu, fnumax=fnumax)
        radius, radius_err =  estimate_radius(numax, dnu, Teff,
                    numax_err=numax_err, dnu_err = dnu_err, Teff_err=Teff_err,
                    fdnu=fdnu, fnumax=fnumax)
        logg, logg_err =  estimate_logg(numax, Teff,
                    numax_err=numax_err, Teff_err=Teff_err,
                    fnumax=fnumax)

        stellar_parameters = {'numax': numax,
                                'numax_err' : numax_err,
                                'dnu' : dnu,
                                'dnu_err' : dnu_err,
                                'mass' : mass,
                                'mass_err' : mass_err,
                                'radius' : radius,
                                'radius_err' : radius_err,
                                'logg' : logg,
                                'logg_err' : logg_err,
                                'Teff' : Teff}
        if Teff_err is not None:
            stellar_parameters['Teff_err'] = Teff_err

        return stellar_parameters

    def autocorrelate(self, numax, width_factor=1):
        """An autocorrelation function for seismic mode envelopes.
        For a given numax, the method calculates the expected Full Width Half
        Maximum of the seismic mode envelope as (Mosser et al 2010)

        fwhm = 0.66 * numax^0.88 .

        Strictly speaking, this is intended for red giants, but will suffice for
        our purposes here for all stars. It then correlates a region of one
        fwhm either side of the estimated numax with itself.

        Before autocorrelating, it also multiplies the section with a hanning
        window, which will increase the autocorrelation power if the region
        has a Gaussian shape, as we'd expect for seismic oscillations.

        Parameters:
        ----------
            numax : float
                The estimated position of the numax of the power spectrum. This
                is used to calculated the region autocorrelated with itself.

            width_factor : float
                This factor is multiplied with the estimated fwhm of the
                oscillation modes, effectively increasing or decreasing the
                autocorrelation range.

        Returns:
        --------
            acf : array-like
                The autocorrelation power calculated for the given numax
        """
        fs = np.median(np.diff(self.frequency))

        if np.isclose(np.median(np.diff(self.frequency.value)), fs.value):
            #Calculate the index FWHM for a given numax
            fwhm = int(np.floor(width_factor * 0.66 * numax.value**0.88 / fs.value)) #Express the FWHM in indices
            fwhm -= fwhm % 2                                    # Make the FWHM value even (%2 = 0 if even, 1 if odd)
            x = int(numax / fs)                                 #Find the index value of numax
            s = np.hanning(len(self.power[x-fwhm:x+fwhm]))      #Define the hanning window for the evaluated frequency space
            C = self.power[x-fwhm:x+fwhm].value * s             #Multiply the evaluated SNR space by the hanning window
            result = np.correlate(C, C, mode='full')            #Correlated the resulting SNR space with itself

        else:
            raise NotImplementedError("The autocorrelate() function requires a grid of evenly spaced frequencies at this time.")

        return result[int(len(result)/2):]      #Return one half of the autocorrelation function

    def _gaussian(self, x, sigma, height, mu):
        """A simple Gaussian function for fitting to autocorrelation peaks."""
        return height * np.exp(-(x - mu)**2 / (2.0 * sigma**2))
