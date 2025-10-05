from Config import Config
from reader import SlidingComplex64Reader
from utils import xp, to_scalar, myfft, optimize_1dfreq_fast, wrap, pltfig, pltfig1
from math import ceil

def fitcoef2(coeff: xp.array, coeftn: xp.array, reader: SlidingComplex64Reader):
    coeflistn = xp.zeros((Config.preamble_len, 3), dtype=xp.float64)

    for pidx in range(0, Config.preamble_len):
        cfo_start = xp.polyval(coeff, pidx)
        bw_start = Config.bw * (1 + cfo_start / Config.sig_freq)
        freq_rate = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi * (1 + 2 * cfo_start / Config.sig_freq)
        # ax^2 + bx + c frequency: (2ax + b)/2pi frequency change rate: a/pi
        coeflistn[pidx, 0] = freq_rate / Config.fs / Config.fs * xp.pi

        tstartn = xp.polyval(coeftn, pidx) # real start time = tstartn + Config.tsign * pidx + reader.tstart
        tendn = xp.polyval(coeftn, pidx + 1) # real end time = tendn + Config.tsign * pidx + reader.tstart
        freq_start = - bw_start * 0.5 + cfo_start
        # (2ax + b)/2pi = freq_start at x = tstartn, b = 2pi(freq_start - 2a tstartn)
        coeflistn[pidx, 1] = 2 * xp.pi * freq_start / Config.fs - 2 * freq_rate * tstartn

        # align 3rd parameter of coef2d to observed phase at tstartn
        nsymbr_start = ceil(tstartn + Config.nsamp / 8 + Config.tsign * pidx)
        nsymbr_end = ceil(tendn - Config.nsamp / 8 + Config.tsign * (pidx + 1))
        nsymbr = xp.arange(nsymbr_start, nsymbr_end)

        sig0 = reader.get(nsymbr_start, nsymbr_end)
        sig1 = sig0 * xp.exp(-1j * xp.polyval(coeflistn[pidx], nsymbr - Config.tsign * pidx))
        data0 = myfft(sig1, n=Config.fft_n, plan=Config.plan)
        freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
        freq, valnew = optimize_1dfreq_fast(sig1, Config.fs, freq1, Config.fs / Config.fft_n * 5)
        coeflistn[pidx, 1] = 2 * xp.pi * (freq_start + freq) / Config.fs - 2 * freq_rate * tstartn
        sig2 = sig0.dot(xp.exp(-1j * xp.polyval(coeflistn[pidx], nsymbr - Config.tsign * pidx)))
        coeflistn[pidx, 2] += xp.angle(sig2)
        print(f"Preamble Symbol {pidx}: C0={coeflistn[pidx,0]:.3e}, C1={coeflistn[pidx,1]:.3e}, C2={coeflistn[pidx,2]:.3e}")

        if pidx < 2:
            nsymbr_start = ceil(tstartn + Config.tsign * pidx)
            nsymbr_end = ceil(tendn + Config.tsign * (pidx + 1))
            sig0 = reader.get(nsymbr_start, nsymbr_end)
            nsymbr = xp.arange(nsymbr_start, nsymbr_end)
            pltfig1(nsymbr, wrap(xp.angle(sig0) - xp.polyval(coeflistn[pidx], nsymbr - Config.tsign * pidx)),
                    title=f"Fitted Phase Curve for Preamble Symbol {pidx}",
                    mode='lines').show()
            pltfig1(nsymbr, xp.angle(sig0 * xp.exp(-1j * xp.polyval(coeflistn[pidx], nsymbr - Config.tsign * pidx))),
                    title=f"Fitted Phase Curve for Preamble Symbol {pidx}",
                    mode='lines').show()
            plt_delta = xp.angle(reader.get(nsymbr_start, nsymbr_start + 1)) - xp.polyval(coeflistn[pidx], nsymbr_start - Config.tsign * pidx)
            pltfig(((nsymbr, xp.unwrap(xp.angle(sig0))), (nsymbr, xp.polyval(coeflistn[pidx], nsymbr - Config.tsign * pidx) + plt_delta)),
                    title=f"Fitted Phase Curve Comparison for Preamble Symbol {pidx}",
                    modes='lines').show()
    return xp.array(coeflistn)

