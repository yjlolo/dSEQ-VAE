python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    train.random_seed=831,277 \
    model.z_dim=8,16,32 \
    model.reg_weights=[0,0,0,0] \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    train.random_seed=736,8 \
    model.z_dim=8,16,32 \
    model.reg_weights=[0,0,0,0] \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    train.random_seed=172,828 \
    model.z_dim=8,16,32 \
    model.reg_weights=[0,0,0,0] \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    train.random_seed=269,750 \
    model.z_dim=8,16,32 \
    model.reg_weights=[0,0,0,0] \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    train.random_seed=98,127 \
    model.z_dim=8,16,32 \
    model.reg_weights=[0,0,0,0] \