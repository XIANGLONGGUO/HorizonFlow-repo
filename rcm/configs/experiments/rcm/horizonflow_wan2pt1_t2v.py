# HorizonFlow: few-step distillation of Wan2.1-T2V-1.3B (paper Sec. 4 and supplementary Tab. S1).
#
# Plain DMD with a fast recipe (student initialized from the teacher, no initialization stage, student learning rate
# 2e-5, one critic update per student update, no sCM loss) plus the two corrections of the DMD pseudo-target:
# Chunk Evenness (ce_*) and Semantic Commitment (sc_*).  600 iterations = 300 student + 300 critic updates.
#
#   torchrun --nproc_per_node=7 -m scripts.train --config=rcm/configs/registry_distill.py -- \
#       experiment=horizonflow_wan2pt1_1pt3B_t2v model.config.fsdp_shard_size=7
#
# or scripts/train_horizonflow.sh, which also sets the checkpoint and data paths.

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyDict

HORIZONFLOW_WAN2PT1_1PT3B_T2V: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/wan2pt1_1pt3B_res480p_t2v_rCM",
            {"override /optimizer": "adamw"},
            {"override /optimizer_fake_score": "adamw"},
            "_self_",
        ],
        job=dict(
            group="HorizonFlow",
            name="horizonflow_wan2pt1_1pt3B_t2v",
            wandb_mode="disabled",
        ),
        optimizer=dict(
            lr=2e-5,  # student; weight decay 0.01, betas (0, 0.999) as in rCM
        ),
        model=dict(
            config=dict(
                optimizer_fake_score=dict(lr=4e-6),  # critic
                loss_scale=0.0,  # no sCM term: pure DMD
                tangent_warmup=0,
                student_update_freq=2,  # alternate critic and student updates (1 : 1)
                teacher_guidance=5.0,
                state_t=21,  # 81 frames at 832 x 480
                # Chunk Evenness
                ce_scale=1.0,
                ce_margin=0.2,
                ce_win=6,
                ce_crop=16,
                ce_steps=4,
                ce_eta=0.03,
                ce_keep=0.95,
                ce_clip=0.5,
                # Semantic Commitment
                sc_scale=1.0,
                sc_tmax=0.9,
                sc_grid=[12, 13],
                sc_cap=0.5,
                sc_time_mean=True,
            )
        ),
        trainer=dict(
            max_iter=600,
            grad_accum_iter=2,  # global batch = number of GPUs x 2 (14 on 7 GPUs)
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1_000_000, run_at_start=False),  # no sampling during training
                every_n_sample_ema=dict(every_n=1_000_000),
            ),
        ),
        checkpoint=dict(
            save_iter=150,
        ),
        dataloader_train=dict(
            num_workers=2,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=HORIZONFLOW_WAN2PT1_1PT3B_T2V["job"]["name"],
    node=HORIZONFLOW_WAN2PT1_1PT3B_T2V,
)
