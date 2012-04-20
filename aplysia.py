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
        ys = self.inst_freq()
        self.ln_f = interpolate.interp1d(xs, ys, **kwds)
        return (self.ln_f(xs))

    def nrm_times(self, anchor):
        self._nrm_times = self.evnt_times - anchor
        return (self._nrm_times)

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

    def before(self, times):
        from numpy import where as where
        selected = times.astype(bool)
        selected[:] = False
        indxs = where( (times<self.on_times[0]) )
        selected[indxs] = True
        return (selected)

    def after(self, times):
        from numpy import where as where
        selected = times.astype(bool)
        selected[:] = False
        indxs = where( (times>self.off_times[-1]) )
        selected[indxs] = True
        return (selected)

    def select(self, rowindxs, subset_str):
        return (on_off_evnts(subset_str,
                             self.on_off_times[rowindxs,:]))

    def _parse_name(self, name):
        self.name = name

class motor_programs(on_off_evnts):
    # need to have a method to iterate over programs
    def __init__(self, prtrct, retrct, **kwds):
        # check types, and add data
        assert issubclass(type(prtrct), on_off_evnts)
        self._prtrct = prtrct

        assert issubclass(type(retrct), on_off_evnts)
        self._retrct = retrct

        self.phs_swtch_times = prtrct.off_times

        # make an array for on_off_times of the motor programs
        from numpy import c_
        mp_times = c_[prtrct.on_times, retrct.off_times]
        super(motor_programs, self).__init__('mps', mp_times, **kwds)

        # for iterating
        self._num_prgs = len(self.on_times)

    def arnd_phs_swtchs(self, times, phs_swtch_width):
        from numpy import where as where
        selected = times.astype(bool)
        selected[:] = False
        for swtch in self.phs_swtch_times:
            indxs = where( (times>swtch - phs_swtch_width) &
                              (times<swtch + phs_swtch_width) )
            selected[indxs] = True
        return (selected)

    def arnd_phs_swtch(self, times, phs_swtch_width):
        from numpy import where as where
        selected = times.astype(bool)
        selected[:] = False
        swtch = self.phs_swtch_times[self._cr_prg]
        indxs = where( (times>swtch - phs_swtch_width) &
                              (times<swtch + phs_swtch_width) )
        selected[indxs] = True
        return (selected)

    def select(self, rowindxs, subset_str):
        return (motor_programs(
                self._prtrct.select(rowindxs, subset_str),
                self._retrct.select(rowindxs, subset_str)))

    def start_time(self):
        return self.on_times[self._cr_prg]

    def phs_swtch_time(self):
        return (self.phs_swtch_times[self._cr_prg])

    def __iter__(self):
        return (self)

    def next(self):
        try:
            self._cr_prg += 1
            if (self._cr_prg+1> self._num_prgs-1):
                raise StopIteration            
        except AttributeError:
            self._cr_prg = 0
        return (self)
    
    def _rewind(self):
        try:
            self.next()
        except StopIteration:
            pass
        self._cr_prg = -1

class experiment(object):
    def __init__(self, **kwds):
        super(experiment, self).__init__(**kwds)

        # for iteration
        self._num_cond = 3

    def add_condition(self, cond):
        assert issubclass(type(cond), on_off_evnts)
        self.expt_cond = cond
        
    def add_cbi2_stims(self, cbi2_stim):
        assert issubclass(type(cbi2_stim), on_off_evnts)
        self.cbi2_stim = cbi2_stim
        self._cbi2_trgd_prgs()

    def add_motor_programs(self, mps):
        assert issubclass(type(mps), motor_programs)
        self.mps = mps

    def add_cell(self, cell):
        assert issubclass(type(cell), ap_cell)
        exec(("self.%s = %s" % (cell.name, 'cell')))

    def _cbi2_trgd_prgs(self):
        trgd_indxs = self.cbi2_stim.with_in(self.mps.on_times)
        self.cbi2_prgs = self.mps.select(trgd_indxs, 'cbi2_triggered')

    def cntrl_prgs(self):
        before_indxs = self.expt_cond.before(self.cbi2_prgs.on_times)
        return (self.cbi2_prgs.select(before_indxs, 'cntrl_cbi2_prgs'))

    def exp_prgs(self):
        exp_indxs = self.expt_cond.with_in(self.cbi2_prgs.on_times)
        return (self.cbi2_prgs.select(exp_indxs, 'exp_cbi2_prgs'))

    def rcv_prgs(self):
        rcv_indxs = self.expt_cond.after(self.cbi2_prgs.on_times)
        return (self.cbi2_prgs.select(rcv_indxs, 'rcv_cbi2_pgrs'))

    def next(self):
        try:
            self._cr_cnd += 1
            if (self._cr_cnd+1> self._num_cond):
                raise StopIteration            
        except AttributeError:
            self._cr_cnd = 0
        if self._cr_cnd == 0:
            return (self.cntrl_prgs())
        if self._cr_cnd == 1:
            return (self.exp_prgs())
        if self._cr_cnd == 2:
            return (self.rcv_prgs())
    
    def _rewind(self):
        try:
            self.next()
        except StopIteration:
            pass
        self._cr_cnd = -1


    def __iter__(self):
        return (self)

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
    expt.add_motor_programs(mps)
    expt.add_cbi2_stims(stims)
    expt.add_condition(conds)
    
    for cell_name in cells.keys():
        tmpcell = ap_cell(cell_name)
        tmpcell.add_spk_times(cells[cell_name])
        expt.add_cell(tmpcell)

    # now getting into plotting and the like, have to split this off somehow.
    import matplotlib.gridspec as gridspec
    nrow = 10
    ncol = 3
    plt_counter = 0
    gs = gridspec.GridSpec(nrow,ncol)

    # plot b48
    # save the bottom row for the averages (row-1)
    from pylab import plt
    for col_num, cond in enumerate(expt):
        for row_num, prog in enumerate(cond):
            ax = plt.subplot(gs[row_num,col_num])
            xs = expt.b48.nrm_times(prog.phs_swtch_time())[1:]
            ys = expt.b48.inst_freq()
            ax.plot(xs, ys)
            ax.set_ylim((0,15))
            ax.set_xlim((-50,20))
            if row_num == nrow - 1:
                break
    plt.show()
    return (expt)

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
        
        
        

    
