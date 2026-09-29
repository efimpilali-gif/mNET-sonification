import os,re,glob,wave,subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter,FuncAnimation
from matplotlib.lines import Line2D

DATA_DIR="."
OUTPUT_FILE="mnet_cosmic_rays_v5.mp4"
WAV_FILE="mnet_cosmic_rays_v5.wav"
FPS=30
SAMPLE_RATE=44100
SHOWER_THRESH=20.0
PITCH_MIN_HZ=180.0
PITCH_MAX_HZ=1600.0
PING_DUR=0.09
DET_COLORS=["#0277BD","#2E7D32","#C62828"]
DET_POS=np.array([[0.00,0.00],[4.20,4.85],[-3.90,4.85]])  # D1, D2, D3 σε μέτρα
# Timing: ~0.032s ανά event (singles) και ~0.1s ανά shower
# Υπολογίζεται αυτόματα μετά τη φόρτωση δεδομένων
DUR_ACT1=50   # placeholder — αντικαθίσταται παρακάτω
DUR_ACT2=50   # placeholder — αντικαθίσταται παρακάτω
TOTAL_DUR=DUR_ACT1+DUR_ACT2
FREEZE_DUR=4
TOP_SHOWERS=5      # showers με pause
PAUSE_DUR=2.0      # δευτερολεπτα pause ανα top shower
TOP1_EXTRA_PAUSE=3.5 # επιπλεον pause για το κορυφαιο event (#1)
BG='#fafaf2';GRID_C='#dddddd'

def load_events(d="."):
    hp=re.compile(r'^(\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$')
    ev=[]
    for fp in sorted(glob.glob(os.path.join(d,"events*.txt"))):
        with open(fp) as f: lines=f.readlines()
        i=0
        while i<len(lines):
            m=hp.match(lines[i].strip())
            if m:
                yr,mo,dy,hr,mn,sc,cnt=[int(x) for x in m.groups()]
                vals=[float(x) for x in lines[i+1].split()] if i+1<len(lines) else []
                peaks=np.array(vals[3:6]) if len(vals)>=6 else np.zeros(3)
                timing=np.array(vals[6:]) if len(vals)>6 else np.zeros(10)
                wfr,j=[],i+2
                while j<len(lines) and len(wfr)<200:
                    p=lines[j].split()
                    if len(p)==3:
                        try: wfr.append([float(x) for x in p])
                        except: break
                    elif hp.match(lines[j].strip()): break
                    j+=1
                wf=np.array(wfr) if wfr else np.zeros((200,3))
                ze=float(timing[3]) if len(timing)>3 else 0.0
                az=float(timing[4]) if len(timing)>4 else 0.0
                ev.append(dict(year=yr,month=mo,day=dy,hour=hr,minute=mn,second=sc,
                               count=cnt,peaks=peaks,timing=timing,waveform=wf,
                               zenith=ze,azimuth=az))
                i=j
            else: i+=1
    ev.sort(key=lambda e:(e['month'],e['day'],e['hour'],e['minute'],e['second']))
    return ev

def pitch(e,emin,emax):
    r=np.clip((e-emin)/(emax-emin+1e-9),0,1)
    return PITCH_MIN_HZ+np.log1p(r*9)/np.log(10)*(PITCH_MAX_HZ-PITCH_MIN_HZ)

def make_ping(freq,dur,amp,pan,reverb=0.0):
    t=np.linspace(0,dur,int(SAMPLE_RATE*dur),endpoint=False)
    s=amp*np.sin(2*np.pi*freq*t)*np.exp(-22*t)
    L=s*np.sqrt((1-pan)/2)
    R=s*np.sqrt((1+pan)/2)
    # Reverb: προσθέτω εξασθενημένο αντίγραφο με καθυστέρηση
    if reverb>0.01:
        delay_samples=int(0.035*SAMPLE_RATE)  # 35ms
        decay=reverb*0.45
        for d,dc in [(delay_samples,decay),(delay_samples*2,decay*0.5),(delay_samples*3,decay*0.25)]:
            if d<len(L):
                L[d:]+=L[:-d]*dc
                R[d:]+=R[:-d]*dc
    return L,R

def make_drone(dur,amp=0.18):
    N=int(dur*SAMPLE_RATE)
    t=np.linspace(0,dur,N,endpoint=False)
    fade=np.ones(N); fade_n=int(3*SAMPLE_RATE)
    fade[:fade_n]=np.linspace(0,1,fade_n)
    sig=np.zeros(N)
    for freq,a in [(55,1.0),(110,0.5),(82,0.3),(165,0.2)]:
        sig+=a*np.sin(2*np.pi*freq*t+np.random.uniform(0,2*np.pi))
    sig=sig/sig.max()*amp*fade
    tremolo=1+0.08*np.sin(2*np.pi*0.7*t)
    sig=sig*tremolo
    return sig,sig*0.95

def make_pause_ping(freq, amp=0.5):
    """Ειδικος ηχος για top shower pause - βαθυ boom"""
    dur=1.2
    t=np.linspace(0,dur,int(SAMPLE_RATE*dur),endpoint=False)
    env=np.exp(-3*t)
    sig=amp*(np.sin(2*np.pi*freq*t)+0.4*np.sin(2*np.pi*freq*0.5*t))*env
    return sig,sig

def build_audio(events,showers,top_shower_ids,frame_timeline):
    """frame_timeline: λιστα (frame, event_idx, act) για σωστο timing"""
    total_audio_dur=(TOTAL_DUR+FREEZE_DUR+TOP_SHOWERS*PAUSE_DUR+TOP1_EXTRA_PAUSE)
    N=int(total_audio_dur*SAMPLE_RATE)
    AL,AR=np.zeros(N),np.zeros(N)
    apk=np.array([e['peaks'] for e in events])
    emin=np.percentile(apk[apk>0],5) if (apk>0).any() else 1
    emax=np.percentile(apk,95)
    n_ev=len(events)

    # Events
    for idx,ev in enumerate(events):
        tc=(idx/max(n_ev-1,1))*(DUR_ACT1-0.5)
        amp=0.15+0.35*(ev['peaks'].mean()/emax)  # ένταση ∝ ενέργεια (peaks mean)
        pan=float(np.sin(np.radians(ev['azimuth'])))
        for det in range(3):
            pk=ev['peaks'][det] if det<len(ev['peaks']) else 0
            if pk<=0: continue
            reverb_ev=float(ev['zenith']/75.0)*0.6  # μεγάλο zenith → περισσότερο reverb
            s0=int((tc+det*0.003)*SAMPLE_RATE); dp=np.clip(pan+(det-1)*0.35,-1,1)
            l,r=make_ping(pitch(pk,emin,emax),PING_DUR,amp*0.6,dp,reverb=reverb_ev)
            e0=min(s0+len(l),N); n=e0-s0
            if n>0 and s0>=0: AL[s0:e0]+=l[:n]; AR[s0:e0]+=r[:n]

    # Drone για Shower Events
    drone_start=int(DUR_ACT1*SAMPLE_RATE)
    drone_dur=DUR_ACT2+FREEZE_DUR+TOP_SHOWERS*PAUSE_DUR+TOP1_EXTRA_PAUSE
    dL,dR=make_drone(drone_dur)
    e0=min(drone_start+len(dL),N); n=e0-drone_start
    if n>0: AL[drone_start:e0]+=dL[:n]; AR[drone_start:e0]+=dR[:n]

    # Shower Events — timing συγχρονισμένο με το video timeline
    # Κάθε shower τοποθετείται γραμμικά στο DUR_ACT2, PLUS οι pauses που έχουν μεσολαβήσει
    n_sh=len(showers)
    for idx,ev in enumerate(showers):
        # Γραμμική θέση μέσα στο act2 (χωρίς pauses)
        t_base=DUR_ACT1+(idx/max(n_sh-1,1))*DUR_ACT2
        # Pauses που έχουν γίνει ΠΡΙΝ από αυτό το shower
        pauses_before=sum(1 for e in showers[:idx] if e in top_shower_ids)
        # Αν το top1 έχει ήδη περάσει, προσθέτω και το extra pause
        top1_passed = any(e is top_shower_ids[-1] for e in showers[:idx])
        t_audio_extra = TOP1_EXTRA_PAUSE if top1_passed else 0.0
        t_audio=t_base+pauses_before*PAUSE_DUR+t_audio_extra
        amp=0.6+0.4*(ev['peaks'].max()/emax)
        pan=float(np.sin(np.radians(ev['azimuth'])))
        for det in range(3):
            pk=ev['peaks'][det] if det<len(ev['peaks']) else 0
            if pk<=0: continue
            f=pitch(pk,emin,emax)
            for hm,ha in [(1.0,1.0),(2.0,0.4),(0.5,0.25)]:
                reverb_sh=float(ev['zenith']/75.0)*0.7
                s0=int((t_audio+det*0.006)*SAMPLE_RATE); dp=np.clip(pan+(det-1)*0.4,-1,1)
                l,r=make_ping(f*hm,PING_DUR*1.5,amp*ha*0.8,dp,reverb=reverb_sh)
                e0=min(s0+len(l),N); n=e0-s0
                if n>0 and s0>=0: AL[s0:e0]+=l[:n]; AR[s0:e0]+=r[:n]

        # Boom για top showers
        if ev in top_shower_ids:
            s0=int(t_audio*SAMPLE_RATE)
            bl,br=make_pause_ping(pitch(ev['peaks'].max(),emin,emax)*0.5, amp=0.7)
            e0=min(s0+len(bl),N); n=e0-s0
            if n>0 and s0>=0: AL[s0:e0]+=bl[:n]; AR[s0:e0]+=br[:n]

        # Extra pause audio για το #1 κορυφαίο event
        if ev is top_shower_ids[-1]:
            t_audio += TOP1_EXTRA_PAUSE  # ο υπολογισμός pauses_before δεν αλλάζει εδώ

    pk=max(np.abs(AL).max(),np.abs(AR).max(),1e-9)
    AL,AR=AL/pk*0.92,AR/pk*0.92
    return np.tanh(AL*1.4)/1.4,np.tanh(AR*1.4)/1.4

def save_wav(AL,AR,fname):
    iv=np.empty(2*len(AL),dtype=np.int16)
    iv[0::2]=(AL*32767).astype(np.int16); iv[1::2]=(AR*32767).astype(np.int16)
    with wave.open(fname,'w') as wf:
        wf.setnchannels(2); wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE); wf.writeframes(iv.tobytes())
    print("WAV: "+fname)

def sax(ax,title="",xlabel="",ylabel="",ylim=None,xlim=None):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor('#aaaaaa')
    ax.tick_params(colors='#555555',labelsize=7)
    if title:  ax.set_title(title, color='#333333',fontsize=8,pad=4)
    if xlabel: ax.set_xlabel(xlabel,color='#555555',fontsize=7)
    if ylabel: ax.set_ylabel(ylabel,color='#555555',fontsize=7)
    if ylim:   ax.set_ylim(*ylim)
    if xlim:   ax.set_xlim(*xlim)
    ax.yaxis.grid(True,color=GRID_C,linewidth=0.6); ax.set_axisbelow(True)

def build_frame_timeline(events, showers, top_shower_ids):
    """
    Κατασκευαζει το timeline των frames:
    - Events: γραμμικα σε DUR_ACT1 δευτερολεπτα
    - Shower Events: γραμμικα σε DUR_ACT2 δευτερολεπτα
      αλλα τα top showers εχουν PAUSE_DUR*FPS extra frames
    Επιστρεφει λιστα: κάθε frame -> (event, act, is_pause)
    """
    timeline=[]
    n_ev=len(events)
    n_sh=len(showers)

    # Φάση 1: Events
    act1_frames=int(DUR_ACT1*FPS)
    for f in range(act1_frames):
        idx=int((f/max(act1_frames-1,1))*(n_ev-1))
        timeline.append((events[min(idx,n_ev-1)],1,False))

    # Praxi 2: Shower Events με pauses στα top
    act2_frames=int(DUR_ACT2*FPS)
    pause_frames=int(PAUSE_DUR*FPS)
    for si in range(n_sh):
        # Frames για αυτο το shower
        f_start=int((si/max(n_sh-1,1))*(act2_frames-1))
        f_end  =int(((si+1)/max(n_sh-1,1))*(act2_frames-1)) if si<n_sh-1 else act2_frames
        n_frames_sh=max(f_end-f_start,1)
        ev=showers[si]
        for f in range(n_frames_sh):
            timeline.append((ev,2,False))
        # Pause για top showers
        if ev in top_shower_ids:
            # Το κορυφαίο event (#1) παίρνει επιπλέον pause
            extra=int(TOP1_EXTRA_PAUSE*FPS) if ev is top_shower_ids[-1] else 0
            for f in range(pause_frames+extra):
                timeline.append((ev,2,True))

    # Freeze
    freeze_frames_n=int(FREEZE_DUR*FPS)
    for f in range(freeze_frames_n):
        timeline.append((showers[-1],3,False))

    return timeline

def create_animation(events,showers,top_shower_ids,output_file,wav_file):
    all_az=np.array([e['azimuth'] for e in events])
    all_ze=np.array([e['zenith']  for e in events])
    sh_az =np.array([e['azimuth'] for e in showers])
    sh_ze =np.array([e['zenith']  for e in showers])

    # Δυναμικά στατιστικά για το sky map label
    n_total   = len(events)
    n_showers = len(showers)
    n_singles = n_total - n_showers
    if len(all_az) > 0:
        mean_az = np.degrees(np.arctan2(
            np.mean(np.sin(np.radians(all_az))),
            np.mean(np.cos(np.radians(all_az)))
        )) % 360
        dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                'S','SSW','SW','WSW','W','WNW','NW','NNW']
        mean_dir_lbl = dirs[int((mean_az+11.25)//22.5) % 16]
        skymap_info = "%d events  |  %d showers  |  mean az: %.0f° (%s)" % (
            n_singles, n_showers, mean_az, mean_dir_lbl)
    else:
        skymap_info = "%d events  |  %d showers" % (n_total, n_showers)
    sh_pk =np.array([e['peaks'].mean() for e in showers])
    sh_en_norm=(sh_pk-sh_pk.min())/(sh_pk.max()-sh_pk.min()+1e-9)

    timeline=build_frame_timeline(events,showers,top_shower_ids)
    total_frames=len(timeline)
    print("Total frames: %d (%.1fs)"%(total_frames,total_frames/FPS))

    apk=np.array([e['peaks'] for e in events])
    emax=np.percentile(apk,98)
    wf_ymax=emax*1.5

    fig=plt.figure(figsize=(16,9),facecolor=BG); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(4,4,left=0.05,right=0.97,top=0.91,bottom=0.07,hspace=0.52,wspace=0.38)
    ax_hdr =fig.add_subplot(gs[0,:])
    ax_map =fig.add_subplot(gs[1:3,0])
    ax_sky =fig.add_subplot(gs[1:3,1],projection='polar')
    ax_wf1 =fig.add_subplot(gs[1,2]); ax_wf2=fig.add_subplot(gs[1,3])
    ax_wf3 =fig.add_subplot(gs[2,2]); ax_nrg=fig.add_subplot(gs[2,3])
    ax_hist=fig.add_subplot(gs[3,0]); ax_info=fig.add_subplot(gs[3,1:])
    for ax in [ax_hdr,ax_map,ax_wf1,ax_wf2,ax_wf3,ax_nrg,ax_hist,ax_info]:
        ax.set_facecolor(BG)

    # HEADER
    ax_hdr.axis('off')
    act_t=ax_hdr.text(0.5,0.75,"Events",transform=ax_hdr.transAxes,
                      ha='center',fontsize=13,fontweight='bold',color='#222222',fontfamily='monospace')
    inf_t=ax_hdr.text(0.5,0.1,"",transform=ax_hdr.transAxes,ha='center',
                      fontsize=9,color='#555555',fontfamily='monospace')

    # DETECTOR MAP
    lim=8
    sax(ax_map,"Detector Layout","x (m, E+)","y (m, N+)",ylim=(-lim,lim),xlim=(-lim,lim))
    ax_map.set_aspect('equal')
    for ang,lbl in [(90,'E'),(0,'N'),(270,'W'),(180,'S')]:
        rad=np.radians(ang)
        ax_map.annotate('',xy=(6.5*np.cos(rad),6.5*np.sin(rad)),xytext=(0,0),
        arrowprops=dict(arrowstyle='->',color='#aaaaaa',lw=0.7,alpha=0.4))
        ax_map.text(7.2*np.cos(rad),7.2*np.sin(rad),lbl,ha='center',va='center',color='#888888',fontsize=7)
    p2,p3=DET_POS[1],DET_POS[2]; v=p3-p2; v_n=v/np.linalg.norm(v)
    s1=p2-v_n*2.0; e1=p3+v_n*1.5
    ax_map.plot([s1[0],e1[0]],[s1[1],e1[1]],color='#FFD600',lw=2.0,alpha=0.4,zorder=1)
    ax_map.text((s1[0]+e1[0])/2-1.0,(s1[1]+e1[1])/2-0.6,'window',
                color='#cc9900',fontsize=6,alpha=0.85,rotation=np.degrees(np.arctan2(v[1],v[0])))
    p1=DET_POS[0]; s2=p1-v_n*2.5; e2=p1+v_n*2.5
    ax_map.plot([s2[0],e2[0]],[s2[1],e2[1]],color='#FFD600',lw=1.2,alpha=0.25,ls='--',zorder=1)
    ax_map.add_patch(plt.Polygon(DET_POS,fill=False,edgecolor='#aaaaaa',lw=0.8,ls='--'))
    cg_list=[]
    for k,(pos,col) in enumerate(zip(DET_POS,DET_COLORS)):
        ax_map.add_patch(plt.Circle(pos,0.55,color=col,alpha=0.12,zorder=2))
        cg=plt.Circle(pos,0.55,color=col,alpha=0.0,zorder=3); ax_map.add_patch(cg); cg_list.append(cg)
        ax_map.text(pos[0],pos[1]+0.95,"D%d"%(k+1),ha='center',color=col,fontsize=8,fontweight='bold')
    s_arrow=ax_map.annotate('',xy=(0.01,0.01),xytext=(0,0),
                             arrowprops=dict(arrowstyle='->',color='#FFD600',lw=2.5),zorder=6)
    s_arrow.set_visible(False)

    # SKY MAP
    ax_sky.set_facecolor('#1a1a2e')
    ax_sky.set_theta_zero_location('N'); ax_sky.set_theta_direction(-1); ax_sky.set_ylim(0,75)
    ax_sky.set_yticks([15,30,45,60,75])
    ax_sky.set_yticklabels(['15','30','45','60','75'],color='#8090b0',fontsize=6)
    ax_sky.set_xticks(np.radians([0,45,90,135,180,225,270,315]))
    ax_sky.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'],color='#a0b0d0',fontsize=8)
    ax_sky.grid(color='#2a2a4a',linewidth=0.7)
    ax_sky.set_title("Sky Map\n(Zenith / Azimuth)",color='#555555',fontsize=8,pad=10)
    th_win=np.radians(np.linspace(80,220,100))
    ax_sky.fill_between(th_win,0,75,color='#FFD600',alpha=0.06,zorder=0)
    ax_sky.plot([np.radians(80)]*2,[0,75],color='#FFD600',lw=0.8,ls=':',alpha=0.5)
    ax_sky.plot([np.radians(220)]*2,[0,75],color='#FFD600',lw=0.8,ls=':',alpha=0.5)
    ax_sky.plot(0,0,'o',color='#ffffff',markersize=3,alpha=0.4,zorder=2)
    sc_n=ax_sky.scatter([],[],s=10,c='#4488ff',alpha=0.45,zorder=3,linewidths=0)
    sc_s=ax_sky.scatter([],[],s=50,c='#FFD600',alpha=0.88,zorder=5,edgecolors='#ffffff',linewidths=0.5)
    sc_c=ax_sky.scatter([],[],s=120,c='#ffffff',alpha=0.95,zorder=6,edgecolors='#FFD600',linewidths=2.0)
    sc_ping=ax_sky.scatter([],[],s=300,c='none',alpha=0.0,zorder=5,edgecolors='#ffffff',linewidths=1.0)
    ax_sky.legend(handles=[
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#4488ff',markersize=5,label='Event',linestyle='None'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#FFD600',markersize=7,label='Shower',linestyle='None')],
        loc='lower right',fontsize=6,facecolor='#1a1a2e',labelcolor='#a0b0d0',framealpha=0.8,edgecolor='#3a3a5a')
    sky_naz,sky_nze,sky_saz,sky_sze=[],[],[],[]

    # WAVEFORMS
    t_wf=np.linspace(0,200,200); wf_ln=[]; wf_fs=[None,None,None]
    for k,(ax,col) in enumerate(zip([ax_wf1,ax_wf2,ax_wf3],DET_COLORS)):
        sax(ax,"D%d Waveform"%(k+1),"sample","mV",ylim=(-3,wf_ymax),xlim=(0,200))
        ln,=ax.plot(t_wf,np.zeros(200),color=col,lw=1.1); wf_ln.append(ln)

    # ENERGY BARS
    sax(ax_nrg,"Peak Amplitude (mV)","","mV",ylim=(0,emax*1.15))
    e_bars=ax_nrg.bar([0,1,2],[0,0,0],color=DET_COLORS,alpha=0.85,edgecolor='#ffffff22',linewidth=0.6,width=0.55)
    ax_nrg.set_xticks([0,1,2]); ax_nrg.set_xticklabels(['D1','D2','D3'],color='#444444',fontsize=8)
    ax_nrg.axhline(SHOWER_THRESH,color='#cc9900',lw=1.0,ls='--',alpha=0.8)
    ax_nrg.text(2.4,SHOWER_THRESH+1,'shower\nthreshold',color='#cc9900',fontsize=6,ha='right')

    # HISTOGRAM
    sax(ax_hist,"Energy Distribution","mV","events")
    ax_hist.hist(apk[apk>0].flatten(),bins=30,color='#4466aa',alpha=0.7,edgecolor='none')
    hmark=ax_hist.axvline(0,color='#FFD600',lw=1.5)

    # INFO
    ax_info.axis('off')
    i_txt=ax_info.text(0.05,0.6,"",transform=ax_info.transAxes,fontsize=10,
                        color='#222222',fontfamily='monospace',va='center')
    sk_txt=ax_info.text(0.55,0.6,"",transform=ax_info.transAxes,fontsize=9,
                         color='#555555',fontfamily='monospace',va='center')

    # PROGRESS BAR
    pr_ax=fig.add_axes([0.05,0.025,0.92,0.012])
    pr_ax.set_facecolor('#dddddd'); pr_ax.axis('off'); pr_ax.set_xlim(0,1); pr_ax.set_ylim(0,1)
    pr_fl=pr_ax.fill_betweenx([0,1],0,0,color='#3355CC',alpha=0.85)
    # Υπολογισμος θεσης διαχωριστη (act1 frames / total frames)
    act1_f=int(DUR_ACT1*FPS)
    sep_x=act1_f/total_frames
    pr_ax.axvline(sep_x,color='#cc9900',lw=0.8,ls=':')
    pr_ax.text(sep_x+0.005,0.5,'Showers',color='#cc9900',fontsize=5,va='center')

    fig.text(0.5,0.965,"mNET  |  Cosmic Ray Sonification & Visualization",
             ha='center',fontsize=14,fontweight='bold',color='#111111',fontfamily='monospace')

    prev_id=[None]; ping_frame=[0]; PING_ANIM=8

    def update(frame):
        ev,act,is_pause=timeline[frame]
        peaks=ev['peaks']; wf=ev['waveform']
        issh=np.all(peaks>SHOWER_THRESH)
        is_top=ev in top_shower_ids

        # FREEZE FRAME
        if act==3:
            act_t.set_text("mNET  |  Total Sky Map")
            act_t.set_color('#cc9900')
            inf_t.set_text(skymap_info)
            inf_t.set_color('#555555')
            sc_n.set_offsets(np.c_[np.radians(all_az),all_ze])
            sc_n.set_sizes([8]*len(all_az))
            sh_sz=40+100*sh_en_norm
            sc_s.set_offsets(np.c_[np.radians(sh_az),sh_ze])
            sc_s.set_sizes(sh_sz)
            sc_c.set_offsets(np.c_[[],[]])
            sc_ping.set_offsets(np.c_[[],[]])
            prog=1.0
            pr_fl.set_paths([np.array([[0,0],[prog,0],[prog,1],[0,1]])])
            return ([act_t,inf_t,sc_n,sc_s,sc_c,sc_ping,pr_fl]
                    +list(e_bars)+wf_ln+cg_list+[s_arrow])

        # HEADER
        if act==1:
            act1_frames_done=sum(1 for t in timeline[:frame+1] if t[1]==1)
            act1_total=sum(1 for t in timeline if t[1]==1)
            pct=int(act1_frames_done/max(act1_total,1)*100)
            act_t.set_text("Events  [%3d%%]"%pct)
            act_t.set_color('#222222')
        else:
            act2_frames_done=sum(1 for t in timeline[:frame+1] if t[1]==2)
            act2_total=sum(1 for t in timeline if t[1]==2)
            pct=int(act2_frames_done/max(act2_total,1)*100)
            if is_top and is_pause:
                # Αναβοσβηνει κατα τη διαρκεια pause
                if (frame//4)%2==0:
                    act_t.set_text("*** TOP SHOWER ***")
                    act_t.set_color('#cc4400')
                else:
                    act_t.set_text("Showers  [%3d%%]"%pct)
                    act_t.set_color('#cc9900')
            elif issh and (frame//4)%2==0:
                act_t.set_text("*** SHOWERS ***  [%3d%%]"%pct)
                act_t.set_color('#cc9900')
            else:
                act_t.set_text("Showers  [%3d%%]"%pct)
                act_t.set_color('#cc9900' if issh else '#222222')

        ts="%d-%02d-%02d  %02d:%02d:%02d"%(ev['year'],ev['month'],ev['day'],
                                             ev['hour'],ev['minute'],ev['second'])
        tag=""
        if is_top: tag="  *** TOP SHOWER ***"
        elif issh: tag="  *** SHOWER ***"
        inf_t.set_text("%s   D1=%.1f  D2=%.1f  D3=%.1f mV%s"%(
            ts,peaks[0],peaks[1],peaks[2],tag))
        inf_t.set_color('#cc4400' if is_top else ('#cc9900' if issh else '#555555'))

        # SKY MAP
        ev_id=id(ev)
        if ev_id!=prev_id[0]:
            az=np.radians(ev['azimuth']); ze=ev['zenith']
            if issh: sky_saz.append(az); sky_sze.append(ze)
            else:    sky_naz.append(az); sky_nze.append(ze)
            prev_id[0]=ev_id
            ping_frame[0]=frame

        if sky_naz: sc_n.set_offsets(np.c_[sky_naz,sky_nze])
        if sky_saz: sc_s.set_offsets(np.c_[sky_saz,sky_sze])
        az_cur=np.radians(ev['azimuth']); ze_cur=ev['zenith']
        sc_c.set_offsets(np.c_[[az_cur],[ze_cur]])
        # Ping animation
        df=frame-ping_frame[0]
        if df<PING_ANIM:
            ping_alpha=1.0-df/PING_ANIM
            ping_size=100+400*(df/PING_ANIM)
            sc_ping.set_offsets(np.c_[[az_cur],[ze_cur]])
            sc_ping.set_sizes([ping_size])
            sc_ping.set_alpha(ping_alpha*0.7)
            c='#cc4400' if is_top else ('#cc9900' if issh else '#333333')
            sc_ping.set_edgecolors([c])
        else:
            sc_ping.set_offsets(np.c_[[],[]])

        n_acc=len(sky_naz)+len(sky_saz)
        # Direction vector 3D
        ze_r=np.radians(ev['zenith']); az_r=np.radians(ev['azimuth'])
        dx=np.sin(ze_r)*np.sin(az_r)
        dy=np.sin(ze_r)*np.cos(az_r)
        dz=np.cos(ze_r)
        sk_txt.set_text("Sky Map: %d events\nZe: %.1f°  Az: %.1f°\n"
                        "dir: (%.2f, %.2f, %.2f)"%(
            n_acc,ev['zenith'],ev['azimuth'],dx,dy,dz))

        # WAVEFORMS
        for k in range(3):
            y=wf[:,k] if wf.shape[0]==200 and wf.shape[1]>k else np.zeros(200)
            wf_ln[k].set_ydata(y)
            if wf_fs[k] is not None: wf_fs[k].remove()
            col_alpha=0.5 if is_top else (0.35 if issh else 0.12)
            wf_fs[k]=[ax_wf1,ax_wf2,ax_wf3][k].fill_between(t_wf,y,color=DET_COLORS[k],alpha=col_alpha)

        # ENERGY BARS
        for k,bar in enumerate(e_bars):
            bar.set_height(max(peaks[k],0)); bar.set_alpha(0.5+0.5*min(peaks[k]/emax,1))

        # HISTOGRAM
        hmark.set_xdata([peaks.mean()])

        # DETECTOR GLOW - εντονοτερο για top showers
        for k in range(3):
            g=min(peaks[k]/emax,1) if k<len(peaks) else 0
            glow_mult=1.5 if is_top else 1.0
            cg_list[k].set_alpha(min(g*0.8*glow_mult,1.0))
            cg_list[k].set_radius(0.55+1.4*g*glow_mult)

        # ΒΕΛΟΣ με μηκος αναλογο zenith
        L_min=0.8; L_max=6.0
        ze_norm=np.clip(ev['zenith']/75.0,0,1)
        L=L_min+ze_norm*(L_max-L_min)
        if ev['azimuth']>0:
            rad=np.radians(ev['azimuth'])
            dx=L*np.sin(rad); dy=L*np.cos(rad)
            s_arrow.set_visible(True)
            s_arrow.xy=(dx,dy); s_arrow.xyann=(0,0)
            col='#cc4400' if is_top else ('#cc9900' if issh else '#3355aa')
            s_arrow.arrowprops['color']=col
        else: s_arrow.set_visible(False)

        # INFO
        i_txt.set_text("Zenith:   %.1f deg\nAzimuth:  %.1f deg"%(ev['zenith'],ev['azimuth']))

        # PROGRESS
        prog=frame/max(total_frames-1,1)
        pr_fl.set_paths([np.array([[0,0],[prog,0],[prog,1],[0,1]])])

        return ([act_t,inf_t,sc_n,sc_s,sc_c,sc_ping,i_txt,sk_txt,hmark,s_arrow,pr_fl]
                +list(e_bars)+wf_ln+cg_list)

    print("Rendering %d frames..."%total_frames)
    anim=FuncAnimation(fig,update,frames=total_frames,interval=1000//FPS,blit=False)
    writer=FFMpegWriter(fps=FPS,metadata={'title':'mNET v5'},
                        extra_args=['-vcodec','libx264','-crf','20','-pix_fmt','yuv420p'])
    tmp=output_file.replace('.mp4','_noaudio.mp4')
    anim.save(tmp,writer=writer,dpi=120); plt.close(fig)
    print("Video (no audio): "+tmp)
    cmd=['ffmpeg','-y','-i',tmp,'-i',wav_file,'-c:v','copy','-c:a','aac','-shortest',output_file]
    r=subprocess.run(cmd,capture_output=True)
    if r.returncode==0: os.remove(tmp); print("Final: "+output_file)
    else: os.rename(tmp,output_file); print("ffmpeg failed: "+output_file)

if __name__=="__main__":
    print("="*55); print("  mNET Cosmic Ray Sonification  v5"); print("="*55)
    events=load_events(DATA_DIR)
    if not events: raise SystemExit("No events found!")
    apk=np.array([e['peaks'] for e in events])
    sm=np.all(apk>SHOWER_THRESH,axis=1)
    showers=[e for e,s in zip(events,sm) if s]
    singles=[e for e,s in zip(events,sm) if not s]
    print("Events: %d  |  Showers: %d"%(len(events),len(showers)))

    # Αυτόματο timing βάσει αριθμού events
    # ~0.032s ανά single event (min 30s, max 90s)
    # ~0.12s ανά shower event (min 20s, max 90s)
    DUR_ACT1 = int(max(30, min(90, len(singles)*0.032)))
    DUR_ACT2 = int(max(20, min(90, len(showers)*0.12)))
    TOTAL_DUR = DUR_ACT1 + DUR_ACT2
    print("DUR_ACT1=%ds  DUR_ACT2=%ds  TOTAL=%.0fs"%(DUR_ACT1,DUR_ACT2,TOTAL_DUR+FREEZE_DUR+TOP_SHOWERS*PAUSE_DUR))

    # Top 5 showers βασει μεσης ενεργειας
    sh_energy=[e['peaks'].mean() for e in showers]
    top_idx=np.argsort(sh_energy)[-TOP_SHOWERS:]
    top_shower_ids=[showers[i] for i in top_idx]
    print("Top %d showers (mean energy): %s mV"%(
        TOP_SHOWERS,[round(sh_energy[i],1) for i in top_idx]))

    print("\nBuilding audio...")
    AL,AR=build_audio(events,showers,top_shower_ids,[])
    save_wav(AL,AR,WAV_FILE)
    create_animation(events,showers,top_shower_ids,OUTPUT_FILE,WAV_FILE)
    print("\nDone!")
    print("  Video: "+OUTPUT_FILE)
    print("  Audio: "+WAV_FILE)
