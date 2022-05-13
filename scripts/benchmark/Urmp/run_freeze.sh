python train.py -m \
    hydra/launcher=joblib \
    data=urmp \
    model=FreezeDsae \
    train.random_seed=831,277 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    data=urmp \
    model=FreezeDsae \
    train.random_seed=736,8 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    data=urmp \
    model=FreezeDsae \
    train.random_seed=172,828 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \