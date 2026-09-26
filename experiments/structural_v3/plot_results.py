import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);a=ap.parse_args()
data=json.loads(a.summary.read_text())
assert data['complete'],'Do not plot partial experimental results as complete'
order=['Free','Greedy','Beam','Critic','Beam+Critic','Critic+Cartesian']
metrics=['SUCR@10','OGR@10','Oracle-RDCR@10','RDCR@10']
colors=['#3272AD','#7CA6D0','#B6B18B','#DDD264'];hatches=['///','\\\\','xx','..']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(10.6,3.9),layout='constrained')
x=np.arange(len(order));width=.195
for i,m in enumerate(metrics):
    y=np.array([data['variants'][v]['percent'][m] for v in order])
    bars=ax.bar(x+(i-1.5)*width,y,width,color=colors[i],edgecolor='black',linewidth=.85,hatch=hatches[i],label=m,zorder=3)
    for j,(bar,value) in enumerate(zip(bars,y)):
        # Offset identical paired values to keep labels readable.
        dy=1.2+(1.8 if i%2 and abs(value-data['variants'][order[j]]['percent'][metrics[i-1]])<1.5 else 0)
        ax.text(bar.get_x()+bar.get_width()/2,value+dy,f'{value:.1f}',ha='center',va='bottom',fontsize=8)
ax.set_xticks(x,['Free','Greedy','Beam','Critic','Beam + Critic','Critic + Cartesian\n(Table 1 Ours)'])
ax.set_ylabel('Score (%)');ax.set_ylim(0,min(105,max(data['variants'][v]['percent'][m] for v in order for m in metrics)+9))
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',linestyle='--',alpha=.3,zorder=0)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=4,frameon=False,fontsize=10)
for suffix in ['png','pdf','svg']:fig.savefig(a.summary.parent/f'structural_ablation_v10_v3.{suffix}',dpi=300,bbox_inches='tight')
print(a.summary.parent/'structural_ablation_v10_v3.png')
