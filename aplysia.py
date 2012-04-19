import pdb
class ap_cell(object):
    def __init__(self, name_str, **kwds):
        super(ap_cell, self).__init__(**kwds)
        self._parse_name(name_str)

    def _parse_name(self, name):
        from mhp_re import bccl_cll_re, cbi_cll_re, crbrl_cll_re
        bccl_name = bccl_cll_re.search(name)
        cbi_name = cbi_cll_re.search(name)
        crbrl_name = crbrl_cll_re.search(name)
        if (bccl_name):
            self.name, self.side = bccl_name.groups()
        elif (cbi_name):
            self.name, self.side = cbi_name.groups()
        elif (crbrl_name):
            self.name, self.side = crbrl_name.groups()
        else:
            raise NameError('Your Aplysia neuron is not named\
how I expect, and I am confused.')

    def add_spk_times(self, times):
        self.evnt_times = times

    def ISIs(self):
        from numpy import diff
        return (diff(self.evnt_times))

    def inst_freq(self):
        return (1/self.ISIs())

    def inst_freq_intrp(self, xs, **kwds):
        # linear interpolation by default
        from scipy import interpolate
        self.ln_f = interpolate.interp1d(xs,self.inst_freq(), **kwds)
        return (self.ln_f(xs))

class on_off_evnts(object):
    def __init__(self, name_str, a_times, **kwds):
        super(on_off_evnts, self).__init__(**kwds)
        self._add_on_off_time(a_times)

    def _add_on_off_time(self, a_times):
        from numpy import transpose
        # check that times have okay dims
        try:
            assert (a_times.shape[1]==2)
        except AssertionError:
            assert (a_times.shape[0]==2)
            a_times = transpose(a_times)
        except AssertionError:
            raise IndexError('the on off time series must have 2 columns')
        self.on_off_times = a_times
        self.on_times = a_times[:,0]
        self.off_times = a_times[:,1]

    def with_in(self, times):
        from numpy import where as where
        selected = times.astype(bool)
        selected[:] = False
        for row in self.on_off_times:
            indxs = where( (times>row[0]) & (times<row[1]) )[0]
            selected[indxs] = True
        return (selected)

    def _parse_name(self, name):
        self.name = name

class motor_programs(on_off_evnts):
    def __init__(self, prtrct, retrct, **kwds):
        # check types, and add data
        assert issubclass(type(prtrct), on_off_evnts)
        self._prtrct = prtrct

        assert issubclass(type(retrct), on_off_evnts)
        self._retrct = retrct

        self._phs_swtch = prtrct.off_times

        # make an array for on_off_times of the motor programs
        from numpy import c_
        mp_times = c_[prtrct.on_times, retrct.off_times]
        super(motor_programs, self).__init__('mps', mp_times, **kwds)

    def around_phs_swtch(self, times, phs_swtch_width):
        from numpy import where as where
        selected = times.astype(bool)
        selected[:] = False
        for swtch in self._phs_swtch:
            indxs = where( (times>swtch - phs_swtch_width) &
                              (times<swtch + phs_swtch_width) )
            selected[indxs] = True
        return (selected)

class experiment(object):
    def __init__(self, **kwds):
        super(experiment, self).__init__(**kwds)

    def add_condition(self, cond):
        assert issubclass(type(cond), on_off_evnts)
        self.expt_cond = cond
        
    def add_cbi2_stims(self, cbi2_stim):
        assert issubclass(type(cbi2_stim), on_off_evnts)
        self.cbi2_stim = cbi2_stim

    def add_motor_programs(self, mps):
        assert issubclass(type(mps), motor_programs)
        self.mps = mps

    def add_cell(self, cell):
        assert issubclass(type(cell), ap_cell)
        exec(("self.%s = %s" % (cell.name, 'cell')))

def main(filename):
    # construct experiment from header
    import spk2_mp
    conds, stims, mps, cells = spk2_mp.main(filename)
    
    conds = on_off_evnts(conds['name'], conds['times'])
    stims = on_off_evnts(stims['name'], stims['times'])

    prtrct = on_off_evnts('prtrct', mps['prtrct'])
    retrct = on_off_evnts('retrct', mps['retrct'])
    mps = motor_programs(prtrct, retrct)

    expt = experiment()
    expt.add_condition(conds)
    expt.add_cbi2_stims(stims)
    expt.add_motor_programs(mps)

    for cell_name in cells.keys():
        tmpcell = ap_cell(cell_name)
        tmpcell.add_spk_times(cells[cell_name])
        expt.add_cell(tmpcell)
    return (expt)

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
        
        
        

    
