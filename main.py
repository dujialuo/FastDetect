# %%
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from find_intersections import find_intersections
from fitcoef import fitcoef2

file_path = "data/test_1226"
reader = SlidingComplex64Reader(file_path, 4240091)

coeff = xp.array((-0.512392321665, -41023.388364708379), dtype=xp.float64)
coeftn = xp.array((0.101716420e-6, 10082.6333 - Config.tsign, 0.37), dtype=xp.float64) 
print(xp.polyval(coeftn, 252))
print(xp.polyval(coeftn, 50))

coeflist = fitcoef2(coeff, coeftn, reader)

sec_xlist = []
sec_tlist = []
if True:
    for pidx in xp.arange(Config.preamble_len):
        tstart2 = xp.polyval(coeftn, pidx) + Config.tsign * pidx
        tdiff = xp.polyval(coeftn, pidx + 1) + Config.tsign * (pidx + 1) - tstart2
        if pidx > 0:
            selected = find_intersections(coeflist[pidx - 1], coeflist[pidx], tstart2, tdiff, reader, 10, pidx - 1, draw=(pidx == 0 or pidx % 50 == 0 or pidx==225)) #!!! TODO remove range
        else:
            nsymbr_start = ceil(xp.polyval(coeftn, pidx) - Config.nsamp / 8 + Config.tsign * pidx)
            nsymbr_end = ceil(xp.polyval(coeftn, pidx) + Config.tsign * pidx)
            nsymbr = xp.arange(nsymbr_start, nsymbr_end)
            sig0 = reader.get(nsymbr_start, nsymbr_end)
            coefstart = xp.hstack(([0., 0.], xp.angle(xp.sum(sig0)))).astype(xp.float64)
            selected = find_intersections(xp.zeros_like(coefstart), coeflist[pidx], tstart2, 0, reader, 10, pidx, draw=True) #!!! TODO remove range
        if selected != None:
            sec_xlist.append(pidx)
            sec_tlist.append(to_scalar(selected))
    sec_xlist = xp.array(sec_xlist)
    sec_tlist = xp.array(sec_tlist)
    with open(f"intersections0.pkl","wb") as f:
        pickle.dump((sec_xlist, sec_tlist), f)

with open(f"intersections0.pkl","rb") as f:
    sec_xlist, sec_tlist = pickle.load(f)

coeff_time = xp.polyfit(sec_xlist, sec_tlist, 1)
print(f"guessed: {coeftn=} coeff_time={coeff_time[0]:.12f},{coeff_time[1]:.12f} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
pltfig(((sec_xlist, sec_tlist), (sec_xlist, xp.polyval(coeff_time, sec_xlist))), title="intersect points fitline").show()
pltfig1(sec_xlist, sec_tlist - xp.polyval(coeff_time, sec_xlist), title="intersect points diff").show()

sec_tdiff_list = sec_tlist - xp.polyval(coeff_time, sec_xlist)
sec_smoothed_tlist = sec_tlist.copy()
for pidx in range(1, len(sec_tdiff_list) - 1):
    if abs(sec_tdiff_list[pidx] - sec_tdiff_list[pidx-1]) > 0.2e-6 and abs(sec_tdiff_list[pidx] + sec_tdiff_list[pidx-1]) > 0.2e-6:
        sec_tdiff_list[pidx] = (sec_tdiff_list[pidx-1] + sec_tdiff_list[pidx+1])/2
        sec_smoothed_tlist[pidx] = (sec_smoothed_tlist[pidx-1] + sec_smoothed_tlist[pidx+1])/2

coeff_time_error = xp.polyfit(sec_xlist, sec_tdiff_list, 1)
pltfig(((sec_xlist, sec_tdiff_list), (sec_xlist, xp.polyval(coeff_time_error, sec_xlist))),
       title="intersect points fit on difference").show()
pltfig1(sec_xlist, sec_tdiff_list - xp.polyval(coeff_time_error, sec_xlist), title="intersect points diff 2").show()
print(f"coeff_time_error={coeff_time_error[0]:.12f},{coeff_time_error[1]:.12f}")


coeff_time31 = xp.polyfit(sec_xlist, sec_smoothed_tlist, 1)
print(f"{coeff_time31=}")

coeff_time3 = xp.polyfit(sec_xlist, sec_smoothed_tlist, 2)
freq_start = (1 - (coeff_time3[0] + coeff_time3[1]) / (2 ** Config.sf / Config.bw)) * Config.sig_freq
freq_rate = - 2 * coeff_time3[0] / (2 ** Config.sf / Config.bw) * Config.sig_freq # frequency change rate, estimated from time change rate

beta = Config.bw / ((2 ** Config.sf) / Config.bw) * (1 + 2 * freq_start / Config.sig_freq)
print("beta=", beta)
sigt = 2 ** Config.sf / Config.bw * (1 + freq_start / Config.sig_freq)
print(f"symbol duration: {sigt} s")
print(f"time diff caused by {coeff_time3[0]=} over symbol duration: { xp.polyval(coeff_time3, Config.preamble_len) - xp.polyval(coeff_time3[1:], Config.preamble_len)} s")
print(f"freq diff caused by time drift over symbol duration: { - beta * (xp.polyval(coeff_time3, Config.preamble_len) - xp.polyval(coeff_time3[1:], Config.preamble_len))} Hz, neglegible")
pltfig(((sec_xlist, xp.polyval(coeff_time3, sec_xlist) - sec_smoothed_tlist), (sec_xlist, xp.polyval(coeff_time31, sec_xlist) - sec_smoothed_tlist)), title=f"intersection fit error 2d/1d").show()
coeff_time3 = xp.hstack([0, coeff_time31[0], coeff_time31[1]]) # ~!!! TODO

pidx_range = xp.arange(Config.preamble_len)
# TODO simplify
estcoefs = []
dd = xp.zeros((2, Config.preamble_len), dtype=xp.float64)
for ixx in range(2):
    print(f"start computing {'start' if ixx == 0 else 'end'} frequencies from coeflist and tjump")
    for pidx in range(Config.preamble_len):
        estf = xp.polyval(coeff, pidx)
        if ixx == 0:
            bwdiff = -Config.bw * (1 + estf / Config.sig_freq) / 2
        else:
            bwdiff = Config.bw * (1 + estf / Config.sig_freq) / 2
        dd[ixx, pidx] = (coeflist[pidx, 0] * 2 * xp.polyval(coeff_time3, pidx + ixx) + coeflist[pidx, 1]) / 2 / xp.pi - bwdiff
pltfig(((pidx_range, dd[0, pidx_range]), (pidx_range, xp.polyval(xp.polyfit(pidx_range, dd[0, pidx_range], 1), pidx_range))),
       title="start intersect points fitline freq0").show()
pltfig1(pidx_range, dd[1, pidx_range] - dd[0, pidx_range], title="start intersect points diff freq0").show()

estt_diff = xp.zeros((Config.preamble_len,), dtype=xp.float64)
for pidx in range(Config.preamble_len):
    estf = xp.polyval(coeff, pidx)
    estt_from_f = (1 - estf / Config.sig_freq) * (2 ** Config.sf / Config.bw)
    estt_diff[pidx] = estt_from_f - (xp.polyval(coeff_time3, pidx) - xp.polyval(coeff_time3, pidx - 1))
pltfig1(pidx_range, estt_diff[pidx_range], title="estimated time difference from freq").show()

# compute coeff_time_final so that estt_from_f = (xp.polyval(coeff_time_final, pidx + 1) - xp.polyval(coeff_time_final, pidx)) for all pidx
def compute_coeff_time_final(coeff_freq, Config, xp):
    c1 = coeff_freq[0]
    c0 = coeff_freq[1]
    common_factor = (2**Config.sf / Config.bw) / Config.sig_freq
    m = -c1 * common_factor
    n = (Config.sig_freq - c0) * common_factor
    a = m / 2.0
    b = n - a  # which is n - m/2
    c = 0.0 
    coeff_time_final = xp.array([to_scalar(a), to_scalar(b), to_scalar(c)], dtype=xp.float64)
    assert a * 2 / (2 ** Config.sf / Config.bw) == -coeff_freq[0] / Config.sig_freq
    assert a + b == (1 - coeff_freq[1] / Config.sig_freq) * (2 ** Config.sf / Config.bw)
    return coeff_time_final

# Compute the final coefficients
coeff_time_final = compute_coeff_time_final(coeff, Config, xp)
coeff_time_final[2] = coeff_time3[2]  # keep the constant term from previous fit
pltfig1(pidx_range, xp.polyval(coeff_time_final, pidx_range) - xp.polyval(coeff_time3, pidx_range), title="final intersect points diff 2d/2d").show()
for pidx in range(Config.preamble_len):
    estf = xp.polyval(coeff, pidx)
    estt_from_f = (1 - estf / Config.sig_freq) * (2 ** Config.sf / Config.bw)
    estt_diff[pidx] = estt_from_f - (xp.polyval(coeff_time_final, pidx) - xp.polyval(coeff_time_final, pidx - 1))
pltfig1(pidx_range, estt_diff[pidx_range], title="estimated time difference from freq").show()
print(f"{coeff_time_final=} {coeff_time3=} {coeff_time_final - coeff_time3=}")
print(f"{coeff_time_final=} {coeftn=} {coeff_time_final - coeftn=}")



for ixx in range(2):
    print(f"start computing {'start' if ixx == 0 else 'end'} frequencies from coeflist and tjump")
    dd = []
    for pidx in range(240):
        estf = xp.polyval(coeff, pidx)
        if ixx == 0:
            bwdiff = -Config.bw * (1 + estf / Config.sig_freq) / 2
        else:
            bwdiff = Config.bw * (1 + estf / Config.sig_freq) / 2
        dd.append(to_scalar((coeflist[pidx, 0] * 2 * xp.polyval(coeff_time_final, pidx + ixx) + coeflist[pidx, 1]) / 2 / xp.pi - bwdiff))
    dd = xp.array(dd)
    pidx_range2 = xp.arange(50, Config.preamble_len - 10)
    estcoef = xp.polyfit(pidx_range2, dd[pidx_range2], 1)
    intercept = xp.mean(dd[pidx_range2] - freq_rate * pidx_range2)
    estcoefs.append(estcoef)

    pltfig(((pidx_range2, dd[pidx_range2]), (pidx_range2, xp.polyval(estcoef, pidx_range2))),
           title=f"intersect points fitline freq{ixx} {estcoef=}").show()
    pltfig1(pidx_range2, dd[pidx_range2] - xp.polyval(estcoef, pidx_range2), title=f"intersect points diff freq{ixx}").show()
    
    fdiff = intercept - estcoef[1] # freq = (2at + b) / 2pi deltaf = a/pi deltat
    tdiff =  fdiff / xp.mean(coeflist[:, 0]) * xp.pi
    print(f"new computation: estcoef at t=0: {estcoef[1]:.12f} estf change rate per symb: {estcoef[0]:.12f} old estimation from tdiff: {freq_rate:.12f} {intercept:.12f} {tdiff:.12f}")

    # f(x) = a x + b
    # f(x + 0.5) = a x + 0.5a + b
    # t(x) = T ( x - f(0)/F - f(1)/F - ... - f(n)/F)
    # t(x) = T ( x - x(f(0)+f(x)/2)/F)
    # t(x) = T ( x - x(ax+b+b)/2/F)
    # t(x) = T (ax^2/2F + x (b/F + 1))
    tsign = 2 ** Config.sf / Config.bw
    assert len(estcoef) == 2
    estcoeft = xp.hstack([estcoef[0] / 2 / Config.sig_freq * tsign, ((estcoef[0] + estcoef[1]) / Config.sig_freq + 1) * tsign, coeftn[-1]])
    print(f"t from f {estcoeft=}, {coeff_time_final=}")
    # pltfig1(pidx_range2, xp.polyval(estcoeft, pidx_range2) - xp.polyval(coeff_time_final, pidx_range2), title="time difference of new and old estimation").show()

    # compute tdiff
    # t(x) = ax^2 + bx + c
    # t(x+1)-t(x) = a(2x+1) + b = 2ax + (a + b)
    assert len(coeff_time_final) == 3
    coeff_tlen = xp.hstack((coeff_time_final[0] * 2, coeff_time_final[0] + coeff_time_final[1]))
    # f from t: t = T * (1 - f / F)
    # f = F * (1 - t / T) 
    # f = F * (1 - (a x + b) / T) = - a x F / T + F * (1 - b / T)
#    coeff_from_tlen = xp.hstack((-  Config.sig_freq / tsign * coeff_tlen[0], Config.sig_freq * (1 - coeff_tlen[1] / tsign)))
    # -2aF/T, F(1- (a+b)/T)
    coeff_from_tlen = xp.hstack([- coeff_time_final[0] * 2 * Config.sig_freq / tsign, Config.sig_freq * (1 - xp.sum(coeff_time_final[:2]) / tsign)])

    print(f"{coeff_tlen=} {coeff_from_tlen=} {estcoef=}") 
    
    

coeff_time = coeff_time_final # todo!!!2
time_delta = 1 / Config.bw

betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi
coeffitlist = xp.zeros((Config.preamble_len, 3), dtype=xp.float64)
coeffitlist[:, 0] = betai * (1 + 2 * xp.polyval(estcoef, pidx_range) / Config.sig_freq)

bwdiff = - Config.bw * (1 + estcoef[1] / Config.sig_freq) / 2
coeffitlist[:, 1] = 2 * xp.pi * xp.polyval(estcoef, pidx_range) - xp.polyval(coeff_time, pidx_range) * 2 * coeffitlist[:, 0] + bwdiff * 2 * xp.pi

for pidx in pidx_range[1:]:
    coeffitlist[pidx, 2] -= wrap(xp.polyval(coeffitlist[pidx], xp.polyval(coeff_time, pidx) - time_delta) - xp.polyval(coeffitlist[pidx - 1], xp.polyval(coeff_time, pidx) - time_delta))

codephase = xp.zeros((Config.total_len,), dtype=xp.float64)
powers = xp.zeros((Config.total_len,), dtype=xp.float64)
codephase_secondary = xp.zeros((Config.total_len,), dtype=xp.float64)

# preamble codephase and powers
for pidx in range(Config.preamble_len):
    x1 = ceil(xp.polyval(coeff_time, pidx) * Config.fs)
    x2 = ceil(xp.polyval(coeff_time, pidx + 1) * Config.fs)
    nsymbr = xp.arange(x1, x2)
    tsymbr = nsymbr / Config.fs
    sig = reader.get(x1, x2 - x1)
    res = sig.dot(xp.exp(-1j * xp.polyval(coeffitlist[pidx], tsymbr)))
    codephase[pidx] = xp.angle(res)
    powers[pidx] = xp.abs(res) / xp.sum(xp.abs(sig))
    if pidx in [0, 120, Config.preamble_len - 1]:
        # coeffitlist_comp = coeffitlist[pidx].copy()
        # coeffitlist_comp[2] += xp.angle(sig[0]) - xp.polyval(coeffitlist[pidx], tsymbr[0])
        print(f"{coeffitlist[pidx]=} {xp.polyval(coeffitlist[pidx], tsymbr[0])=} {xp.unwrap(xp.angle(sig))[-1]=} {xp.polyval(coeffitlist[pidx], tsymbr[-1]) - xp.polyval(coeffitlist[pidx], tsymbr[0]) + xp.angle(sig[0])=}")
        pltfig((
            (tsymbr, xp.unwrap(xp.angle(sig)) ), 
            (tsymbr, xp.polyval(coeffitlist[pidx], tsymbr) - xp.polyval(coeffitlist[pidx], tsymbr[0]) + xp.angle(sig[0]))), title=f"preamble codephase {pidx=} angle={xp.angle(res)} pow={xp.abs(res)/xp.sum(xp.abs(sig))}").show()
        pltfig1(tsymbr, xp.unwrap(xp.angle(sig)) - xp.polyval(coeffitlist[pidx], tsymbr) + xp.polyval(coeffitlist[pidx], tsymbr[0]) - xp.angle(sig[0]), title=f"preamble codephase {pidx=} fitdiff").show()
pltfig1(xp.arange(Config.preamble_len), xp.unwrap(codephase[:Config.preamble_len]), title="unwrap phase").show()
pltfig1(xp.arange(Config.preamble_len), powers[:Config.preamble_len], title="powers").show()

fig=None

pidx = -1
x1 = math.ceil(xp.polyval(coeff_time, pidx) * Config.fs)
x2 = math.ceil(xp.polyval(coeff_time, pidx + 2) * Config.fs)
print(x1, x2)
nsymbr = xp.arange(x1, x2)
sig = reader.get(x1, x2 - x1)
pltfig1(None, xp.unwrap(xp.angle(sig)), title = "Pidx=-1 and Pidx 0").show()


coeffitlist = xp.concatenate((coeffitlist, xp.zeros_like(coeffitlist)), axis=0) # 2, 240, 3, 2
coeffitlist = xp.stack((coeffitlist, xp.zeros_like(coeffitlist)), axis=0) # 2, 240, 3, 2
print(coeffitlist.shape)
coeffitlist[1, :, -1] = coeffitlist[0, :, -1] # share the last value as continuous phase


startphase = 0#xp.polyval(coeffitlist[Config.preamble_len + 4], xp.polyval(coeff_time, Config.preamble_len + 5 - 0.75))

codexdiffs = []
codes = xp.zeros((Config.total_len,), dtype=int)
for pidx in range(Config.sfdend, Config.total_len):
    tstart = xp.polyval(coeff_time, pidx - 0.75)
    tend = xp.polyval(coeff_time, pidx + 1 - 0.75)
    x1 = math.ceil(tstart * Config.fs)
    x2 = math.ceil(tend * Config.fs)
    nsymbr = xp.arange(x1, x2)
    sig = reader.get(x1, x2 - x1)
    if xp.mean(xp.abs(sig)) < 0.01:
        print(f"{pidx=} {xp.mean(xp.abs(sig))=} too small. is symbol ending? quitting, payload_len={pidx - Config.sfdend}")
        break
    
    x1 = math.ceil(tstart * Config.fs)
    x2 = math.ceil(tend * Config.fs)
    nsymbr = xp.arange(x1, x2)
    tsymbr = nsymbr / Config.fs

    # pltfig1(tsymbr, xp.unwrap(xp.angle(reader.get(x1, x2-x1))), title=f"{pidx=}").show()
    # pltfig1(tsymbr, xp.abs(reader.get(x1, x2-x1)), title=f"{pidx=}").show()
    assert xp.mean(xp.abs(reader.get(x1, x2-x1))) > 0.1, f"{pidx=} {xp.mean(xp.abs(reader.get(x1, x2-x1)))=} too small. is symbol ending?"
    estcoef_this = xp.polyval(coeff, pidx)

    beta1 = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi * (1 + 2 * estcoef_this / Config.sig_freq)
    estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
    beta2 = 2 * xp.pi * (xp.polyval(coeff, pidx) - estbw / 2) - tstart * 2 * beta1  # 2ax+b=differential b=differential - 2 * beta1 * time
    coef2d_est = xp.array([to_scalar(beta1), to_scalar(beta2), 0])

    sig2 = reader.get(x1, x2-x1) * xp.exp(-1j * xp.polyval(coef2d_est, tsymbr))
    data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
    freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
    freq, valnew = optimize_1dfreq_fast(sig2, tsymbr, freq1, Config.fs / Config.fft_n * 5) # valnew may be as low as 0.3, only half the power will be collected
    # freq = freq1 # todo !!!
    # assert valnew > 0.3, f"{freq=} {freq1=} {valnew=}"
    if freq < 0: freq += estbw
    codex = freq / estbw * 2 ** Config.sf
    code = around(codex)
    # print(f"{codex=} {code=}")

    tmid = tstart * (code / 2 ** Config.sf) + tend * (1 - code / 2 ** Config.sf)
    tmid = tmid.item()
    x3 = math.ceil(tmid * Config.fs)

    nsymbr1 = xp.arange(x1, x3)
    tsymbr1 = nsymbr1 / Config.fs
    nsymbr2 = xp.arange(x3, x2)
    tsymbr2 = nsymbr2 / Config.fs

    beta2 = (2 * xp.pi * (xp.polyval(coeff, pidx) + estbw * (code / 2 ** Config.sf - 0.5))
             - tstart * 2 * beta1)
    coef2d_est2 = xp.array([to_scalar(beta1), to_scalar(beta2), 0])
    coef2d_est2_2d = xp.polyval(coef2d_est2, tstart) - startphase
    coef2d_est2[2] -= coef2d_est2_2d

    beta2a = (2 * xp.pi * (xp.polyval(coeff, pidx) + estbw * (code / 2 ** Config.sf - 1.5))
              - tstart * 2 * beta1)
    coef2d_est2a = xp.array([to_scalar(beta1), to_scalar(beta2a), 0])
    coef2d_est2a_2d = xp.polyval(coef2d_est2a, tmid) - xp.polyval(coef2d_est2, tmid)
    coef2d_est2a[2] -= coef2d_est2a_2d

    res2 = reader.get(x1, x3-x1).dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr1))) / xp.sum(xp.abs(reader.get(x1, x3-x1)))
    res2a = reader.get(x3, x2-x3).dot(xp.exp(-1j * xp.polyval(coef2d_est2a, tsymbr2))) / xp.sum(xp.abs(reader.get(x3, x2-x3)))

    if not (xp.abs(res2).item() > 0.7 or code > 2 ** Config.sf * 0.7) or not (xp.abs(res2a).item() > 0.7 or code < 2 ** Config.sf * 0.2):
        pltfig1(tsymbr1, xp.angle(reader.get(x1, x3-x1) * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr1))), title=f"{pidx=} 1st angle {codex=} pow={xp.abs(res2).item()}").show()
        pltfig1(tsymbr2, xp.angle(reader.get(x3, x2-x3) * xp.exp(-1j * xp.polyval(coef2d_est2a, tsymbr2))), title=f"{pidx=} 2st angle {codex=} pow={xp.abs(res2a).item()}").show()

    assert xp.abs(res2).item() > 0.7 or code > 2 ** Config.sf * 0.7, f"{pidx=} {code=} 1st power {xp.abs(res2).item()}<0.7"
    assert xp.abs(res2a).item() > 0.7 or code < 2 ** Config.sf * 0.2, f"{pidx=} {code=} 2nd power {xp.abs(res2a).item()}<0.7"

    endphase = xp.polyval(coef2d_est2a, tend)
    ifreq1 = 2 * xp.pi * (xp.polyval(coeff, pidx) + estbw * (code / 2 ** Config.sf - 0.5))
    ifreq2 = 2 * xp.pi * (xp.polyval(coeff, pidx) + estbw * (code / 2 ** Config.sf - 1.5))

    # startphase = endphase
    # powers.append(xp.abs(res2).item())
    # powers.append(xp.abs(res2a).item())
    # codephase2.append(xp.angle(res2).item())
    # codephase2.append(xp.angle(res2a).item())
    # codephase.append(xp.angle(res2).item())
    # codephase.append(xp.angle(res2a).item())
    # coef2d_ests.append(coef2d_est2)
    # coef2d_ests.append(coef2d_est2a)
    codes[pidx] = code
    codexdiffs.append(abs(codex - code))
pltfig1(None, xp.unwrap(codexdiffs), title="codexdiffs").show()

coef_f = estcoefs[0].copy()
# coef_f[-1] -= 16 # coef is also changing with time
coef_t = coeff_time3
codes[Config.preamble_len] = 8
codes[Config.preamble_len + 1] = 16 ## TODO
pidx_delta = 0.75
betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi # frequency slope to phase 2d slope, *pi

for pidx in range(Config.total_len):
 
    code = codes[pidx]
    if pidx >= Config.sfdend:
        cfo_start = xp.polyval(coef_f, pidx - pidx_delta)
    else:
        cfo_start = xp.polyval(coef_f, pidx)

    bw_start = Config.bw * (1 + cfo_start / Config.sig_freq)

    if True:# pidx >= Config.sfdend:
        tstart_delta = - 1 / bw_start
    else:
        tstart_delta = 0.0

    if pidx >= Config.sfdend:
        tstart = xp.polyval(coef_t, pidx - pidx_delta) 
    else:
        tstart = xp.polyval(coef_t, pidx)
    if pidx >= Config.sfdend - 1: 
        tend = xp.polyval(coef_t, pidx + 1 - pidx_delta) 
    else:
        tend = xp.polyval(coef_t, pidx + 1)

    if pidx >= Config.sfdpos and pidx < Config.sfdend:
        coef1_x2 = - betai * (1 + 2 * cfo_start / Config.sig_freq) # coef1_x2 = bw * pi
        coef1_x = 2 * xp.pi * (xp.polyval(coef_f, pidx) + bw_start * (code / Config.n_classes + 0.5)) - tstart * 2 * coef1_x2 # freq at tstart = polyval(coef_f, pidx) - bw_start/2 + bw_start * (code / 2^sf-0.5)
    else:
        coef1_x2 = betai * (1 + 2 * cfo_start / Config.sig_freq) # coef1_x2 = bw * pi
        coef1_x = 2 * xp.pi * (xp.polyval(coef_f, pidx) + bw_start * (code / Config.n_classes - 0.5)) - tstart * 2 * coef1_x2 # freq at tstart = polyval(coef_f, pidx) - bw_start/2 + bw_start * (code / 2^sf-0.5)
    coef1 = xp.array([to_scalar(coef1_x2), to_scalar(coef1_x), 0])
    coef1_const = xp.polyval(coef1, tstart + tstart_delta) - xp.polyval(coeffitlist[1, pidx - 1], tstart + tstart_delta)
    coef1[2] -= coef1_const
    coeffitlist[0, pidx] = coef1 # continuing the last phase in coeffitlist
    
    # 2nd part
    # freq at tstart = polyval(coef_f, pidx) - bw_start/2 + bw_start * (code / 2^sf-0.5) - bw_start
    if pidx in range(Config.preamble_len, Config.sfdpos) or pidx >= Config.sfdend:
        if pidx >= Config.sfdend:
            tjump = xp.polyval(coef_t, pidx + 1 - code / Config.n_classes - pidx_delta)
        else:
            tjump = xp.polyval(coef_t, pidx + 1 - code / Config.n_classes)
        cfo_jump = xp.polyval(coef_f, pidx + 1 - code / Config.n_classes)
        bw_jump = Config.bw * (1 + cfo_jump / Config.sig_freq)
        coef2_x2 = coef1_x2 
        coef2_x = coef1_x - 2 * xp.pi * bw_start # 2ax+b=differential b=differential - 2 * coef1_x2 * time
        coef2 = xp.array([to_scalar(coef1_x2), to_scalar(coef2_x), 0])

        assert tstart < tjump < tend, f"{pidx=} {tstart=} {tjump=} {tend=}"
        coef2_const = xp.polyval(coef2, tjump) - xp.polyval(coeffitlist[0, pidx], tjump)
        coef2[2] -= coef2_const
        coeffitlist[1, pidx] = coef2
    else:
        coeffitlist[1, pidx] = coeffitlist[0, pidx]
        # coeffitlist[1, pidx, 0] = 0
        # coeffitlist[1, pidx, 1] = 0
        # coeffitlist[1, pidx, 2] = xp.polyval(coeffitlist[0, pidx], tend) # keep continuous phase

    # print(pidx, xp.polyval(coeffitlist[0, pidx], tstart))
    # print(pidx, xp.polyval(coeffitlist[1, pidx], tend))


    if pidx < Config.sfdend:
        x1 = math.ceil(xp.polyval(coef_t, pidx) * Config.fs)
        x2 = math.ceil(xp.polyval(coef_t, pidx + 1) * Config.fs)
        x3 = math.ceil(xp.polyval(coef_t, pidx + (1 - code / 2 ** Config.sf)) * Config.fs)
    else:
        x1 = math.ceil(xp.polyval(coef_t, pidx - pidx_delta) * Config.fs)
        x2 = math.ceil(xp.polyval(coef_t, pidx + 1 - pidx_delta) * Config.fs)
        x3 = math.ceil(xp.polyval(coef_t, pidx + (1 - code / 2 ** Config.sf) - pidx_delta) * Config.fs)
    if pidx == Config.sfdend - 1:
        x1 = math.ceil(xp.polyval(coef_t, pidx) * Config.fs)
        x2 = math.ceil(xp.polyval(coef_t, pidx + 1 - pidx_delta) * Config.fs)
        x3 = x2
        
    nsymbr1 = xp.arange(x1, x3)
    tsymbr1 = nsymbr1 / Config.fs
    sig1 = reader.get(x1, x3 - x1)
    res1 = sig1.dot(xp.exp(-1j * xp.polyval(coef1, tsymbr1)) )

    codephase[pidx] = xp.angle(res1)
    powers[pidx] = xp.abs(res1) / xp.sum(xp.abs(sig1))
    # print(f"{pidx=} 1st part {code=} {xp.angle(res1)=} pow={xp.abs(res1)/xp.sum(xp.abs(sig1))}")
    # if pidx in [0, 120, 239, 242, 243, 244]:
    if False:# pidx == Config.sfdend - 1:
        pltfig1(tsymbr1, xp.angle(sig1 * xp.exp(-1j * xp.polyval(coef1, tsymbr1))), title=f"residue {pidx=}").show()

        pltfig((
            (tsymbr1, xp.unwrap(xp.angle(sig1))), 
            (tsymbr1, xp.polyval(coef1, tsymbr1) - xp.polyval(coef1, tsymbr1[0]) + xp.angle(sig1[0])),
            ),
            title=f"preamble code {pidx=} {code=} fit curve coef1").show()


        pltfig1(tsymbr1, xp.polyval(coef1, tsymbr1) - xp.polyval(coef1, tsymbr1[0]) + xp.angle(sig1[0]) - xp.unwrap(xp.angle(sig1)),
                 title=f"preamble code {pidx=} {code=} fit curve coef1").show()

    
    if pidx in range(Config.preamble_len, Config.sfdpos) or pidx >= Config.sfdend:
        nsymbr2 = xp.arange(x3, x2)
        tsymbr2 = nsymbr2 / Config.fs
        sig2 = reader.get(x3, x2 - x3)
        res2 = sig2.dot(xp.exp(-1j * xp.polyval(coef2, tsymbr2)) )
        codephase_secondary[pidx] = wrap(xp.angle(res1) - xp.angle(res2))
        powers[pidx] = (xp.abs(res1) + xp.abs(res2)) / (xp.sum(xp.abs(sig1)) + xp.sum(xp.abs(sig2)))

        tsymbrA = xp.arange(x1, x2) / Config.fs
        sigA = reader.get(x1, x2 - x1)
        # print(f"{pidx=} 2nd part {code=} {xp.angle(res2)=} pow={xp.abs(res2)/xp.sum(xp.abs(sig2))}")

        # if pidx in [240, 241] or pidx in range(Config.sfdend, Config.sfdend + 2):
        if False:#pidx in range(Config.sfdend, Config.sfdend + 3):
            pltfig1(tsymbr1, xp.angle(sig1 * xp.exp(-1j * xp.polyval(coef1, tsymbr1))), title=f"residue {pidx=}").show()
            pltfig(((tsymbr1, xp.angle(sig1 * xp.exp(-1j * xp.polyval(coef1, tsymbr1)))),
                (tsymbr2, xp.angle(sig2 * xp.exp(-1j * xp.polyval(coef2, tsymbr2))))), title=f"residue {pidx=}").show()

            pltfig((
                (tsymbrA, xp.unwrap(xp.angle(sigA))), 
                (tsymbr1, xp.polyval(coef1, tsymbr1) - xp.polyval(coef1, tsymbr1[0]) + xp.angle(sig1[0])),
                (tsymbr2, xp.polyval(coef2, tsymbr2) - xp.polyval(coef1, tsymbr1[0]) + xp.angle(sig1[0])),
                ),
                title=f"preamble code {pidx=} {code=} fit curve coef1").show()


            # pltfig((
            #     (tsymbr1, xp.polyval(coef1, tsymbr1) - xp.polyval(coef1, tsymbr1[0]) + xp.angle(sig1[0]) - xp.unwrap(xp.angle(sigA))[:len(tsymbr1)]),
            #     (tsymbr2, xp.polyval(coef2, tsymbr2) - xp.polyval(coef2, tjump) + xp.polyval(coef1, tjump) - xp.polyval(coef1, tsymbr1[0]) + xp.angle(sig1[0]) - xp.unwrap(xp.angle(sigA))[len(tsymbr1):]),
            #     ),
            #     title=f"preamble code {pidx=} {code=} fit curve coef1").show()
    if True:# pidx >= Config.sfdend - 1:
        coeffitlist[0, pidx, 2] += xp.angle(res1)
        if pidx >= Config.sfdend or pidx in range(Config.preamble_len, Config.sfdpos):
            coeffitlist[1, pidx, 2] += xp.angle(res2)
        else:
            coeffitlist[1, pidx, 2] += xp.angle(res1)
            # print(f"Adjusting phase offset at {pidx=}: {xp.angle(res1)} {xp.angle(res2)} {xp.angle(sig1.dot(xp.exp(-1j * xp.polyval(coeffitlist[0, pidx], tsymbr1))) )}, {xp.angle(sig2.dot(xp.exp(-1j * xp.polyval(coeffitlist[1, pidx], tsymbr2)) ))}")
        # else:
                # if pidx >= Config.sfdend + 1 and pidx < Config.sfdend + 20:
        # selected = find_intersections(coeffitlist[0, pidx], coeffitlist[1, pidx - 1], xp.polyval(coef_t, pidx - pidx_delta), reader, 10, pidx, draw=True)
    if pidx in range(Config.sfdend + 5, Config.sfdend + 10):
        print(f"Adjusting phase offset at {pidx=}: {xp.angle(res1)} {xp.angle(sig1[-1].dot(xp.exp(-1j * xp.polyval(coeffitlist[0, pidx], tsymbr1[-1]))) )} {wrap(xp.angle(sig1[-1]) - xp.polyval(coeffitlist[0, pidx], tsymbr1[-1]))}")
        pltfig1(tsymbr1, xp.angle(sig1 * xp.exp(-1j * xp.polyval(coeffitlist[0, pidx], tsymbr1))), title=f"residue {pidx=}").show()
        selected = find_intersections(coeffitlist[0, pidx], coeffitlist[1, pidx - 1], xp.polyval(coef_t, pidx), reader, 3e-5, pidx, draw=True)

pltfig(((xp.arange(Config.total_len), xp.unwrap(codephase[:Config.total_len])), (xp.arange(Config.preamble_len, Config.total_len), xp.unwrap(codephase_secondary[Config.preamble_len : Config.total_len]))), title="preamble+data unwrap phase").show()
pltfig1(xp.arange(Config.total_len ), powers[:Config.total_len ], title="preamble+data powers").show()
pltfig1(None, codes, title="preamble+data codes").show()

coefdiff = xp.zeros((Config.total_len,), dtype=xp.float64)
for pidx in range(1, Config.total_len):
    coefdiff[pidx] = (xp.unwrap(codephase[:Config.total_len])[pidx] - xp.unwrap(codephase[:Config.total_len])[pidx - 1]) / (xp.polyval(coeff_time_final, pidx) - xp.polyval(coeff_time_final, pidx - 1)) / (2 * xp.pi)
line_coef = xp.polyfit(xp.arange(Config.preamble_len // 2, Config.total_len), coefdiff[Config.preamble_len // 2 :], 3)
pltfig(((xp.arange(Config.total_len), coefdiff), (xp.arange(Config.total_len), xp.polyval(line_coef, xp.arange(Config.total_len)))), title="coefdiff").show()
# for pidx in range(Config.preamble_len, Config.total_len):
#     coefdiff[pidx] = xp.polyval(line_coef, pidx)

# %%
