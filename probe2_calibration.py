import os,sys; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2"); os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
sys.path.insert(0,"src"); sys.path.insert(0,".")
import yaml, numpy as np, tensorflow as tf
from upair5g.builders import build_pusch_transmitter, build_channel, build_ls_estimator, get_resource_grid, extract_true_dmrs_mask_per_stream
from upair5g.estimator import UPAIRChannelEstimator
from upair5g.evaluation import _make_eval_batch
from upair5g.utils import complex_sq_abs
VARIANT, USERS, EBNOS, NB = sys.argv[1], int(sys.argv[2]), [float(x) for x in sys.argv[3].split(",")], int(sys.argv[4])
run=f"TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/{VARIANT}"
cfg=yaml.safe_load(open(f"{run}/artifacts/resolved_config.yaml"))
tf.keras.utils.set_random_seed(12345)
tx,_=build_pusch_transmitter(cfg,num_users=USERS); ch=build_channel(cfg,tx); rg=get_resource_grid(tx)
ls=build_ls_estimator(tx,cfg,interpolation_type="lin")
est=UPAIRChannelEstimator(ls_estimator=ls,resource_grid=rg,cfg=cfg,pilot_mask=extract_true_dmrs_mask_per_stream(tx,rg))
wb=_make_eval_batch(tx=tx,channel=ch,cfg=cfg,batch_size=4,ebno_db=EBNOS[0])
est.estimate_with_ls(wb["y"],wb["no"],training=False); est.load_weights(f"{run}/checkpoints/best.weights.h5")
print(f"[PROBE2] variant={VARIANT} u={USERS} ckpt loaded")
for e in EBNOS:
    S=dict(p=0.,ls=0.,up=0.,ep=0.,er=0.); P=[];R=[]
    for b in range(NB):
        bt=_make_eval_batch(tx=tx,channel=ch,cfg=cfg,batch_size=32,ebno_db=e)
        h_hat,err_hat,h_ls,_=est.estimate_with_ls(bt["y"],bt["no"],training=False)
        h=tf.convert_to_tensor(bt["h"])
        se=tf.cast(complex_sq_abs(h-h_hat),tf.float32); sl=tf.cast(complex_sq_abs(h-tf.convert_to_tensor(h_ls)),tf.float32)
        p=tf.cast(complex_sq_abs(h),tf.float32); eh=tf.cast(err_hat,tf.float32)
        S["p"]+=float(tf.reduce_sum(p)); S["ls"]+=float(tf.reduce_sum(sl)); S["up"]+=float(tf.reduce_sum(se))
        S["ep"]+=float(tf.reduce_sum(eh)); S["er"]+=float(tf.reduce_sum(se))
        P.append(tf.reshape(eh,[-1]).numpy()[::7]); R.append(tf.reshape(se,[-1]).numpy()[::7])
    P=np.concatenate(P); R=np.concatenate(R); idx=np.argsort(P)
    q=[float(P[s].mean()/max(R[s].mean(),1e-30)) for s in np.array_split(idx,4)]
    nls,nup=S["ls"]/S["p"],S["up"]/S["p"]
    print(f"[PROBE2-CAL] ebno={e:+.1f} NMSE_ls={nls:.5f} NMSE_upair={nup:.5f} gain_dB={10*np.log10(nls/nup):.2f} "
          f"calib_overall={S['ep']/max(S['er'],1e-30):.3f} calib_by_errhat_quartile={[round(x,3) for x in q]}")
