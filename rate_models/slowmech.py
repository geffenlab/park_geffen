# a testing ground for a possible super slow mechanism to unify results
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt

heav = np.heaviside


T = 30000
dt = 1
t = np.linspace(0,T-dt,int(T/dt))



def dan(x,tl=1):
    return np.exp(-tl*x)

def gt(x,ton,toff):
    return heav(x-ton,1)*heav(toff-x,1)*dan(x-ton)


#################### SSA
t0=100;dur=100;isi_ssa=300

t_on_ssa = np.zeros(50)
t_off_ssa = np.zeros(len(t_on_ssa))

t_on_ssa[0] = t0
t_off_ssa[0] = t_on_ssa[0]+dur

stim_ssa = np.zeros(len(t))

stim_ssa += gt(t,t_on_ssa[0],t_off_ssa[0])

for i in range(1,len(t_on_ssa)):
    t_on_ssa[i] = t_off_ssa[i-1]+isi_ssa
    t_off_ssa[i] = t_on_ssa[i]+dur

    j = np.argmin(np.abs(t-t_on_ssa[i]))
    stim_ssa[j:]+= gt(t[j:],t_on_ssa[i],t_off_ssa[i])


def i_ssa(x):
    x = np.mod(x,T)
    return interp1d(t,stim_ssa)(x)


########################## FS
#p t0_fs=10,dur_fs=5,gap_fs=2,isi_fs=38

t0=100;dur_fs=50;gap_fs=20;isi_fs=380

ta_on_fs = np.zeros(50)
ta_off_fs = np.zeros(len(ta_on_fs))

tb_on_fs = np.zeros(len(ta_on_fs))
tb_off_fs = np.zeros(len(ta_on_fs))

ta_on_fs[0] = t0
ta_off_fs[0] = ta_on_fs[0]+dur_fs
tb_on_fs[0] = ta_off_fs[0]+gap_fs
tb_off_fs[0] = tb_on_fs[0]+dur_fs

stim_fs = np.zeros(len(t))
stim_fs += gt(t,ta_on_fs[0],ta_off_fs[0])
stim_fs += gt(t,tb_on_fs[0],tb_off_fs[0])

for i in range(1,len(ta_on_fs)):
    
    ta_on_fs[i] = tb_off_fs[i-1]+isi_fs
    ta_off_fs[i] = ta_on_fs[i]+dur_fs
    tb_on_fs[i] = ta_off_fs[i]+gap_fs
    tb_off_fs[i] = tb_on_fs[i]+dur_fs

    j1 = np.argmin(np.abs(t-ta_on_fs[i]))
    j2 = np.argmin(np.abs(t-tb_on_fs[i]))
    
    stim_fs[j1:]+= gt(t[j1:],ta_on_fs[i],ta_off_fs[i])
    stim_fs[j2:]+= gt(t[j2:],tb_on_fs[i],tb_off_fs[i])

def i_fs(x):
    x = np.mod(x,T)
    return interp1d(t,stim_fs)(x)


############################### TCA
#p iti_tca=240,isi_tca=30,dur_tca=10

t0=100;iti_tca=2400;isi_tca=300;dur_tca=100

t_on_tca = np.zeros(50)
t_off_tca = np.zeros(len(t_on_tca))

t_on_tca[0] = t0
t_off_tca[0] = t_on_tca[0]+dur_tca

stim_tca = np.zeros(len(t))
stim_tca += gt(t,t_on_tca[0],t_off_tca[0])

for i in range(1,len(t_on_tca)):

    if i%8==0:
    #if i%5==0:
        add = iti_tca
    else:
        add = 0
        
    t_on_tca[i] = t_off_tca[i-1]+isi_tca + add
    t_off_tca[i] = t_on_tca[i]+dur_tca
    #tb_on_tca[i] = ta_off_tca[i]+isi_tca
    #tb_off_tca[i] = tb_on_tca[i]+dur_tca

    j1 = np.argmin(np.abs(t-t_on_tca[i]))
    
    stim_tca[j1:]+= gt(t[j1:],t_on_tca[i],t_off_tca[i])

def i_tca(x):
    x = np.mod(x,T)
    return interp1d(t,stim_tca)(x)

################################## PV act
t0=100;isi_pv=950;dur_pv=50

t_on_pv = np.zeros(50)
t_off_pv = np.zeros(len(t_on_pv))

t_on_pv[0] = t0
t_off_pv[0] = t_on_pv[0]+dur_pv

stim_pv = np.zeros(len(t))
stim_pv += gt(t,t_on_pv[0],t_off_pv[0])

for i in range(1,len(t_on_pv)):
        
    t_on_pv[i] = t_off_pv[i-1]+isi_pv
    t_off_pv[i] = t_on_pv[i]+dur_pv
    #tb_on_pv[i] = ta_off_pv[i]+isi_pv
    #tb_off_pv[i] = tb_on_pv[i]+dur_pv

    j1 = np.argmin(np.abs(t-t_on_pv[i]))
    
    stim_pv[j1:]+= gt(t[j1:],t_on_pv[i],t_off_pv[i])

def i_pv(x):
    x = np.mod(x,T)
    return interp1d(t,stim_pv)(x)



def rhs(y,t,taud1=1500,taud2=100):
    # originally taud1=1500,taud2=100
    gSSA = y[0];gFS=y[1];gTCA=y[2];gPV=y[3]
    
    # thalamic input modifiers, slow timescale depression. Into all neurons
    return np.array([-gSSA**2/taud1+i_ssa(t)/taud2,
                     -gFS**2/taud1+i_fs(t)/taud2,
                     -gTCA**2/taud1+i_tca(t)/taud2,
                     -gPV**2/taud1+i_pv(t)/taud2,
    ])


sol = np.zeros((len(t),4))


for i in range(1,len(t)):
    sol[i,:] = sol[i-1,:]+dt*rhs(sol[i-1,:],t[i])

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(t,i_ssa(t))
#ax.plot(t,i_fs(t))
ax.plot(t,sol)
plt.show()
