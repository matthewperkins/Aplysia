import pdb
class ap_cell(object):
    def __init__(self, name_str, **kwds):
        super(ap_cell, self).__init__(**kwds)
        self._parse_name(name_str)
        self._offset = 0

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

    def inst_freq_intrp(self, xs, kind = 'zero', **kwds):
        # zero interpolation by default, this creates a 'skyline'
        # plot with frequencssy
        # NB drop the last spike time, and pair the firts ISI with
        # the first spike, this plays well weth the 'zero' interp
        from scipy import interpolate
        from numpy import where, zeros, r_
        ys = self.inst_freq()
        self.ln_f = interpolate.interp1d(self.evnt_times[0:-1], ys,
                                         kind = kind, **kwds)

        # trim the new xs to zero if outside range:
        l,h = (self.evnt_times[0], self.evnt_times[-2])
        pre = xs[ where(xs<l) ]
        post = xs[ where(xs>h) ]
        xs = xs[where ((xs>l) & (xs<h)) ]
        interp_ys = self.ln_f(xs)
        # to keep the length the same, add back zeros outside of range
        ys = r_[zeros(len(pre)), interp_ys, zeros(len(post))]
        return (ys)
            
    def nrm_times(self, offset):
        # to un-norm times, have to call again with self._offset*-1
        self._offset = offset
        self.evnt_times -= self._offset
        return (self.evnt_times)

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
        try:
            del self._cr_prg
        except AttributeError:
            pass
        return (self)

    def next(self):
        try:
            self._cr_prg += 1
            if (self._cr_prg > (self._num_prgs-1)):
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
        try:
            del self._cr_cnd
        except AttributeError:
            pass
        return (self)

def quick_plot(expt, prog, cell_str, ax, **kwds):
    from matt_axes_cust import clean_axes
    from numpy import r_
    if 'ylim' in kwds.keys():
        ylim = kwds.pop('ylim')
    else: ylim = (0,15)
    if 'xlim' in kwds.keys():
        xlim = kwds.pop('xlim')
    else: xlim = (-40,20)
    clean_axes(ax, **kwds)
    xs = r_[xlim[0]:xlim[1]:0.4]
    prog_swtch_time = prog.phs_swtch_time()
    # norm spike times
    exec("expt.%s.nrm_times(prog_swtch_time)[1:]" % (cell_str))
    exec("ys = expt.%s.inst_freq_intrp(xs)" % (cell_str))
    # un-norm spike time
    exec("expt.%s.nrm_times(prog_swtch_time*-1)[1:]" % (cell_str))
    ax.plot(xs, ys)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

def main(filename, ylim = (0,15), xlim = (-40,20)):
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
    # figure out max num prgs in any one condition
    max_prgs = max([cnd._num_prgs for cnd in expt])
    import matplotlib.gridspec as gridspec
    nrow = max_prgs
    ncol = 3
    plt_counter = 0
    gs = gridspec.GridSpec(nrow,ncol)

    # plot b48
    # save the bottom row for the averages (row-1)
    from pylab import plt
    for col_num, cond in enumerate(expt):
        if col_num == 0: left_most = True
        else: left_most = False
        for row_num, prog in enumerate(cond):
            if row_num == (cond._num_prgs - 1): bottom_most = True
            else: bottom_most = False
            ax = plt.subplot(gs[row_num,col_num])
            quick_plot(expt, prog, 'b48', ax,
                       left_most = left_most, bottom_most = bottom_most,
                       ylim = ylim, xlim = xlim)
            if row_num == nrow - 1:
                break
    expt.b48_fig = plt.gcf()
    expt.b48_fig.set_size_inches((7.5,10))
    
    # plot both b48 and b8
    # save the bottom row for the averages (row-1)
    for col_num, cond in enumerate(expt):
        if col_num == 0: left_most = True
        else: left_most = False
        for row_num, prog in enumerate(cond):
            if row_num == (cond._num_prgs - 1): bottom_most = True
            else: bottom_most = False
            ax = plt.subplot(gs[row_num,col_num])
            quick_plot(expt, prog, 'b48', ax,
                       left_most = left_most, bottom_most = bottom_most,
                       ylim = ylim, xlim = xlim)
            quick_plot(expt, prog, 'b8', ax,
                       left_most = left_most, bottom_most = bottom_most,
                       ylim = ylim, xlim = xlim)
            # if row_num == nrow:
            #     break
    expt.b8_b48_fig = plt.gcf()
    expt.b8_b48_fig.set_size_inches((7.5,10))
    return (expt)

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
        
        
        

    
