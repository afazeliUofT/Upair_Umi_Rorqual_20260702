from __future__ import annotations
import json, re
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path.cwd().resolve(); OUT=ROOT/'BLER_3u_eval-results'; OUT.mkdir(exist_ok=True)
UR=ROOT/'_isolated_eval_chunks'; BR=ROOT/'_final_u3_baseline_chunks'
U=3; E=[-4.,-3.,-2.,-1.,0.,1.]; TARGET=100; MAXB=2000
V=['main_d256_b4_r2','shallow_d256_b2_r2','deep_d256_b6_r2','narrow_d192_b4_r2','wide_d320_b4_r2','wide_deep_d320_b6_r2','mlpwide_d256_b4_r4']
B=['baseline_ls_lmmse','baseline_ls_2dlmmse_lmmse','perfect_csi_lmmse']
L={
V[0]:'UPAIR main ($d$=256, $L$=4, $r$=2)',V[1]:'UPAIR shallow ($d$=256, $L$=2, $r$=2)',
V[2]:'UPAIR deep ($d$=256, $L$=6, $r$=2)',V[3]:'UPAIR narrow ($d$=192, $L$=4, $r$=2)',
V[4]:'UPAIR wide ($d$=320, $L$=4, $r$=2)',V[5]:'UPAIR wide-deep ($d$=320, $L$=6, $r$=2)',
V[6]:'UPAIR MLP-wide ($d$=256, $L$=4, $r$=4)',B[0]:'LS estimator + LMMSE detector',
B[1]:'LS + 2D-LMMSE estimator + LMMSE detector',B[2]:'Perfect CSI + LMMSE detector'}
S={V[0]:'Main',V[1]:'Shallow',V[2]:'Deep',V[3]:'Narrow',V[4]:'Wide',V[5]:'Wide-deep',V[6]:'MLP-wide',B[0]:'LS',B[1]:'LS+2D-LMMSE',B[2]:'Perfect CSI'}
M=['o','s','^','v','D','P','X']; BM={B[0]:'<',B[1]:'>',B[2]:'*'}; LS={B[0]:'--',B[1]:'-.',B[2]:':'}
K=['variant','receiver','num_users','ebno_db']; SUM=['bit_errors','num_bits','block_errors','num_blocks','num_batches_run']
plt.rcParams.update({'font.family':'serif','font.size':10.5,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42})

def norm(d):
 d=d.copy()
 for c in ['variant','receiver']:
  if c not in d:d[c]=''
  d[c]=d[c].astype(str)
 for c in ['num_users','ebno_db','chunk_idx']+SUM:
  if c not in d:d[c]=np.nan
  d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=K)

def gridmask(s):
 x=pd.to_numeric(s,errors='coerce').to_numpy(float); q=np.zeros(len(x),bool)
 for e in E:q|=np.isclose(x,e,atol=1e-9,rtol=0)
 return q

def finish(d,src):
 if d.empty:return d
 for c in SUM:d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0).round().astype('int64')
 d['ber']=np.where(d.num_bits>0,d.bit_errors/d.num_bits,np.nan); d['bler']=np.where(d.num_blocks>0,d.block_errors/d.num_blocks,np.nan)
 d['reliable']=d.block_errors>=TARGET; d['done']=((d.num_batches_run>=20)&d.reliable)|(d.num_batches_run>=MAXB); d['source']=src
 return d

def raw(root,receivers):
 a=[]
 if not root.exists():return pd.DataFrame()
 for p in root.rglob('chunk_result.csv'):
  try:d=norm(pd.read_csv(p))
  except Exception:continue
  if d.empty:continue
  if d.chunk_idx.isna().all():
   m=re.search(r'chunk(\d+)',str(p)); d['chunk_idx']=int(m.group(1)) if m else hash(str(p))
  d['_mt']=p.stat().st_mtime; a.append(d)
 if not a:return pd.DataFrame()
 d=pd.concat(a,ignore_index=True,sort=False); d=d[d.receiver.isin(receivers)&d.num_users.eq(U)&gridmask(d.ebno_db)].copy()
 if d.empty:return d
 d.chunk_idx=d.chunk_idx.fillna(-1).astype(int); d['_nb']=pd.to_numeric(d.num_blocks,errors='coerce').fillna(-1)
 d=d.sort_values(K+['chunk_idx','_mt','_nb']).drop_duplicates(K+['chunk_idx'],keep='last')
 g=d.groupby(K,as_index=False)[SUM].sum(); return finish(g,'raw chunks')

def merged(root,receivers):
 a=[]
 if not root.exists():return pd.DataFrame()
 for p in root.glob('merged_*.csv'):
  try:d=norm(pd.read_csv(p))
  except Exception:continue
  if d.empty:continue
  d['_mt']=p.stat().st_mtime; a.append(d)
 if not a:return pd.DataFrame()
 d=pd.concat(a,ignore_index=True,sort=False); d=d[d.receiver.isin(receivers)&d.num_users.eq(U)&gridmask(d.ebno_db)].copy()
 if d.empty:return d
 d['_nb']=pd.to_numeric(d.num_blocks,errors='coerce').fillna(-1)
 d=d.sort_values(K+['_nb','_mt']).drop_duplicates(K,keep='last')
 return finish(d[K+SUM].copy(),'merged CSV')

def combine(a,b):
 z=[]
 if not a.empty:a=a.copy();a['_p']=0;z.append(a)
 if not b.empty:b=b.copy();b['_p']=1;z.append(b)
 if not z:return pd.DataFrame(columns=K+SUM+['bler','ber','done','reliable'])
 d=pd.concat(z,ignore_index=True,sort=False); d['_nb']=pd.to_numeric(d.num_blocks,errors='coerce').fillna(-1)
 return d.sort_values(K+['_p','_nb'],ascending=[1,1,1,1,1,0]).drop_duplicates(K,keep='first').drop(columns=['_p','_nb'])

up=combine(raw(UR,{'upair5g_lmmse'}),merged(UR,{'upair5g_lmmse'})); up=up[up.variant.isin(V)&up.receiver.eq('upair5g_lmmse')]
ba=combine(raw(BR,set(B)),merged(BR,set(B))); ba=ba[ba.receiver.isin(B)&ba.variant.eq(V[0])]
d=pd.concat([up,ba],ignore_index=True,sort=False).sort_values(['receiver','variant','ebno_db'])

tr=[]; rr=ROOT/'TWC_plots_comprehensive/runs_rx16/seed7/1dmrs'
for v in V:
 p=rr/v/'metrics/train_state.json'; q=rr/v/'metrics/model_summary.json'; x={};y={}
 try:x=json.loads(p.read_text())
 except Exception:pass
 try:y=json.loads(q.read_text())
 except Exception:pass
 tr.append({'variant':v,'training_complete':bool(x.get('training_complete',False)),'latest_step':x.get('latest_step',-1),'total_steps':x.get('total_steps',40000),'save_reason':x.get('save_reason','missing'),'best_val':x.get('best_val',np.nan),'num_trainable_params':y.get('num_trainable_params',np.nan)})
tr=pd.DataFrame(tr); tr.to_csv(OUT/'training_status.csv',index=False)

look={(r.variant,r.receiver,float(r.ebno_db)):r for r in d.itertuples(index=False)}; st=[]
for v,r,fam in [(v,'upair5g_lmmse','UPAIR') for v in V]+[(V[0],r,'benchmark') for r in B]:
 for e in E:
  x=look.get((v,r,e))
  if x is None: state='missing'; done=False; rel=False; bl=np.nan;er=nb=blocks=0
  else:
   er=int(x.block_errors);nb=int(x.num_batches_run);blocks=int(x.num_blocks);bl=float(x.bler);done=bool(x.done);rel=er>=TARGET
   state='partial' if not done else ('reliable' if rel else ('zero_at_cap' if er==0 else 'capped_below_target'))
  st.append({'family':fam,'variant':v,'receiver':r,'ebno_db':e,'state':state,'done':done,'reliable':rel,'bler':bl,'block_errors':er,'num_blocks':blocks,'num_batches_run':nb})
st=pd.DataFrame(st); st.to_csv(OUT/'evaluation_status_all_expected_points.csv',index=False); st[~st.done].to_csv(OUT/'missing_or_incomplete_points.csv',index=False)
d.to_csv(OUT/'all_aggregated_results_including_zero.csv',index=False); d[d.bler.gt(0)].to_csv(OUT/'publication_positive_bler_rows.csv',index=False); d[d.bler.eq(0)].to_csv(OUT/'zero_bler_points_omitted_from_plots.csv',index=False)

nt=int(tr.training_complete.sum()); nu=int(st[st.family.eq('UPAIR')].done.sum()); nb=int(st[st.family.eq('benchmark')].done.sum())
lines=[f'Training complete: {nt}/7',f'UPAIR points complete: {nu}/42',f'Benchmark points complete: {nb}/18',f'All requested points complete: {nu+nb}/60','Completion: >=100 block errors after >=20 batches, or 2000-batch cap.','BLER=0 is retained in CSV but omitted from every BLER plot.','','INCOMPLETE/MISSING:']
for x in st[~st.done].itertuples(index=False): lines.append(f"{S[x.variant] if x.receiver=='upair5g_lmmse' else S[x.receiver]:18s} {x.ebno_db:+g} dB  {x.state:8s}  errors={x.block_errors} batches={x.num_batches_run}")
(OUT/'evaluation_audit.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))

z=d[d.bler.gt(0)].copy(); z['series']=np.where(z.receiver.eq('upair5g_lmmse'),z.variant.map(S),z.receiver.map(S))
if not z.empty:
 t=z.pivot_table(index='series',columns='ebno_db',values='bler',aggfunc='first').reindex([S[x] for x in V+B]).reindex(columns=E); t.to_csv(OUT/'publication_bler_table.csv')

def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=350 if ext=='png' else None)
 plt.close(fig)

def wilson(k,n,z=1.95996398454):
 p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return np.maximum(c-h,np.finfo(float).tiny),np.minimum(c+h,1)

def curves(keys,name,title,ci=False):
 fig,ax=plt.subplots(figsize=(12.4,7.2)); plotted=0
 for i,(v,r) in enumerate(keys):
  x=d[d.variant.eq(v)&d.receiver.eq(r)].sort_values('ebno_db'); mp={float(a.ebno_db):a for a in x.itertuples(index=False)}
  yy=np.array([float(mp[e].bler) if e in mp and np.isfinite(mp[e].bler) and mp[e].bler>0 else np.nan for e in E])
  if not np.isfinite(yy).any():continue
  base=r!='upair5g_lmmse'; key=r if base else v; color=f'C{7+B.index(r)}' if base else f'C{V.index(v)}'; marker=BM[r] if base else M[V.index(v)]; style=LS[r] if base else '-'
  ax.plot(E,yy,style,color=color,lw=2.15 if base else 1.9,label=L[key],zorder=2)
  p=x[x.bler.gt(0)&np.isfinite(x.bler)].copy(); xx=p.ebno_db.to_numpy(float); y=p.bler.to_numpy(float); rel=p.block_errors.to_numpy()>=TARGET; done=p.done.to_numpy(bool); cap=done&~rel; part=~done
  if ci and len(p):
   lo,hi=wilson(p.block_errors.to_numpy(float),p.num_blocks.to_numpy(float)); ax.errorbar(xx,y,yerr=np.vstack([y-lo,hi-y]),fmt='none',ecolor=color,elinewidth=.85,capsize=2,alpha=.65)
  if rel.any():ax.scatter(xx[rel],y[rel],marker=marker,s=55,color=color,zorder=4)
  if cap.any():ax.scatter(xx[cap],y[cap],marker=marker,s=60,facecolors='none',edgecolors=color,lw=1.5,zorder=4)
  if part.any():ax.scatter(xx[part],y[part],marker='x',s=62,color=color,lw=1.5,zorder=5)
  plotted+=1
 ax.set_yscale('log');ax.set_xticks(E);ax.set_xlim(-4.2,1.2);ax.set_xlabel(r'$E_b/N_0$ (dB)');ax.set_ylabel('Block error rate (BLER)');ax.set_title(title+f'\n{U} active users');ax.grid(True,which='both',ls=':',alpha=.75)
 if plotted:ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=True)
 ax.text(.01,.012,'Filled: ≥100 block errors; open: 2000-batch cap with <100 errors; ×: partial. BLER=0 and unavailable points are omitted.',transform=ax.transAxes,fontsize=8,va='bottom',bbox={'boxstyle':'round,pad=.28','facecolor':'white','edgecolor':'.75','alpha':.9})
 fig.tight_layout();save(fig,name)

uk=[(v,'upair5g_lmmse') for v in V]; bk=[(V[0],r) for r in B]
curves(uk+bk,'Fig1_all_7_upair_and_3_benchmarks_u3','Extended UPAIR architecture variants and receiver benchmarks')
curves(uk,'Fig2_all_7_upair_variants_u3','Extended UPAIR architecture comparison')
curves([(V[0],'upair5g_lmmse')]+bk,'Fig3_main_upair_vs_benchmarks_with_95CI_u3','Main Extended UPAIR versus receiver benchmarks (95% Wilson intervals)',True)

base=d[d.receiver.eq(B[1])&d.bler.gt(0)].set_index('ebno_db'); fig,ax=plt.subplots(figsize=(10.8,6.6));n=0
for i,v in enumerate(V):
 x=d[d.variant.eq(v)&d.receiver.eq('upair5g_lmmse')&d.bler.gt(0)].set_index('ebno_db'); c=sorted(set(x.index.astype(float))&set(base.index.astype(float)))
 if not c:continue
 ax.plot(c,[base.loc[e,'bler']/x.loc[e,'bler'] for e in c],marker=M[i],color=f'C{i}',label=L[v]);n+=1
if n:
 ax.axhline(1,color='.25',ls='--',lw=1.2);ax.set_yscale('log');ax.set_xticks(E);ax.set_xlabel(r'$E_b/N_0$ (dB)');ax.set_ylabel('BLER gain over LS+2D-LMMSE\n(baseline BLER / UPAIR BLER)');ax.set_title(f'Extended UPAIR gain relative to LS+2D-LMMSE\n{U} active users; positive common points only');ax.grid(True,which='both',ls=':',alpha=.75);ax.legend(loc='center left',bbox_to_anchor=(1.02,.5));fig.tight_layout();save(fig,'Fig4_bler_gain_over_2dlmmse_u3')
else:plt.close(fig)

keys=uk+bk; labels=[S[v] for v in V]+[S[r] for r in B]; val={'missing':0,'partial':1,'capped_below_target':2,'zero_at_cap':2,'reliable':3}; sym={'missing':'M','partial':'P','capped_below_target':'C','zero_at_cap':'C','reliable':'R'}
a=np.zeros((len(keys),len(E)),int); q=np.full(a.shape,'M',object)
for i,(v,r) in enumerate(keys):
 for j,e in enumerate(E):
  x=st[st.variant.eq(v)&st.receiver.eq(r)&np.isclose(st.ebno_db,e)].iloc[0];a[i,j]=val[x.state];q[i,j]=sym[x.state]
fig,ax=plt.subplots(figsize=(9.5,6.2));ax.imshow(a,aspect='auto',cmap=ListedColormap(['#f2f2f2','#f4b183','#ffe699','#a9d18e']),vmin=-.5,vmax=3.5);ax.set_xticks(range(len(E)),[f'{e:+g}' for e in E]);ax.set_yticks(range(len(labels)),labels);ax.set_xlabel(r'$E_b/N_0$ (dB)');ax.set_title(f'Evaluation coverage and Monte-Carlo reliability\n{U} active users')
for i in range(a.shape[0]):
 for j in range(a.shape[1]):ax.text(j,i,q[i,j],ha='center',va='center',fontsize=9,fontweight='bold')
h=[Line2D([0],[0],marker='s',color='none',markerfacecolor=c,markeredgecolor='.5',markersize=10,label=t) for c,t in [('#a9d18e','R: ≥100 block errors'),('#ffe699','C: completed at cap, <100 errors'),('#f4b183','P: partial'),('#f2f2f2','M: missing')]];ax.legend(handles=h,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2);fig.tight_layout();save(fig,'Fig5_evaluation_coverage_u3')

(OUT/'README.txt').write_text('3 active users; Eb/N0=-4..+1 dB. UPAIR data are aggregated from _isolated_eval_chunks; benchmarks from _final_u3_baseline_chunks. Raw chunks override merged CSVs. BLER=0/NaN/unavailable points are never plotted. Filled markers: >=100 block errors; open markers: completed at 2000-batch cap with <100 errors; x: partial. Each figure is saved as PNG, PDF and SVG.\n')
print(f'[OK] Results written to {OUT}')
