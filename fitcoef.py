from Config import Config
from reader import SlidingComplex64Reader
from utils import xp, to_scalar, myfft, optimize_1dfreq_fast, wrap, pltfig
from math import ceil
def fitcoef2(coeff: xp.array, coeft: xp.array, reader: SlidingComplex64Reader):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi # frequency slope to phase 2d slope, *pi
    coeflist = []

    for pidx in range(0, Config.preamble_len):
        
        # compute coef2d_est2: polynomial curve fitting unwrapped phase of symbol pidx
        # time: tstart to tend
        # frequency at tstart: - estbw * 0.5 + estf
        estf = xp.polyval(coeff, pidx)
        estbw = Config.bw * (1 + estf / Config.sig_freq)
        beta1 = betai * (1 + 2 * estf / Config.sig_freq)
        tstart = xp.polyval(coeft, pidx)
        tend = xp.polyval(coeft, pidx + 1)
        beta2 = 2 * xp.pi * (- estbw * 0.5 + estf) - tstart * 2 * beta1
        coef2d_est2 = xp.array([to_scalar(beta1), to_scalar(beta2), 0])

        # align 3rd parameter of coef2d_est2 to observed phase at tstart
        nsymbr_start = ceil(tstart * Config.fs + Config.nsamp / 8)
        nsymbr_end = ceil(tend * Config.fs - Config.nsamp / 8)
        nsymbr = xp.arange(ceil(tstart * Config.fs + Config.nsamp / 8), ceil(tend * Config.fs - Config.nsamp / 8))
        tsymbr = nsymbr / Config.fs

        sig0 = reader.get(nsymbr_start, nsymbr_end - nsymbr_start)
        sig1 = sig0 * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        data0 = myfft(sig1, n=Config.fft_n, plan=Config.plan)
        freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
        freq, valnew = optimize_1dfreq_fast(sig1, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # freqf, valnew = optimize_1dfreq(sig1, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # print(f"Initial freq offset: {freq1}, after FFT fit: {freq}, after precise fit: {freqf}, diff: {freqf - freq}")

        # adjust coef2d_est2[1] according to freq difference
        coef2d_est2[1] = 2 * xp.pi * (- estbw * 0.5 + estf + freq) - tstart * 2 * beta1
        sig2 = sig0 * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        # freq, valnew = optimize_1dfreq(sig2, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # print(f"{freq=} should be zero {valnew=}")
        coef2d_est2[2] += xp.angle(sig0.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))))
        # print(f"{xp.angle(sig0.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))))} should be zero")
        coeflist.append(coef2d_est2)

        if False:#pidx < 2:
            pltfig1(tsymbr, wrap(xp.angle(sig0) - xp.polyval(coef2d_est2, tsymbr)),
                    title=f"Fitted Phase Curve for Preamble Symbol {pidx}",
                    mode='lines').show()
            pltfig(((tsymbr, xp.unwrap(xp.angle(sig0))), (tsymbr, xp.polyval(coef2d_est2, tsymbr) - xp.polyval(coef2d_est2, tsymbr[0]) + xp.angle(sig0[0]))),
                    title=f"Fitted Phase Curve Comparison for Preamble Symbol {pidx}",
                    modes='lines').show()
    return xp.array(coeflist)

