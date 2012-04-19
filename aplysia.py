class ap_cell(object):
    def __init__(self, name_str, **kwds):
        super(ap_cell, self).__init__(**kwds)
        self._parse_name(name_str)

    def _parse_name(self, name):
        from mhp_re import bccl_cll_re, crbrl_cll_re
        bccl_name = bccl_cll_re.search(name)
        crbrl_name = crbrl_cll_re.search(name)
        if (bccl_name):
            self.name, self.side = bccl_name.groups()
        elif (crbrl_name):
            self.name, self.side = crbrl_name.groups()
        else:
            raise NameError('Your Aplysia neuron is not named\
how I expect, and I am confused.')

    def _add_spk_times(self, times):
        self.evnt_times = times

    def ISIs(self):
        return (np.diff(self.evnt_times))

    def inst_freq(self):
        return (1/self.ISIs())

    def inst_freq_intrp(self, xs, **kwds):
        # linear interpolation by default
        from scipy import interpolate
        self.ln_f = interpolate.interp1d(xs,self.inst_freq(), **kwds)
        return (self.ln_f(xs))

class expt_on_off(object):
    def __init__(self, name_str, **kwds):
        super(expt_on_off, self).__init__(**kwds)

    def _add_on_off_time(self, a_times):
        # check that times have okay dims
        try:
            assert (a_times.shape[1]==2)
        except AssertionError:
            assert (a_times.shape[0]==2)
            a_times = np.transpose(a_times)
        except AssertionError:
            raise IndexError('the on off time series must have 2 columns')
        self.on_off_times = a_times
        self.on_times = a_times[:,0]
        self.off_times = a_times[:,1]

    def _parse_name(self, name):
        
        

    
