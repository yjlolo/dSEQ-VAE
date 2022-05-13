python train.py -m \
    hydra/launcher=joblib \
    train.random_seed=831,277 \
    data=urmp \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    train.random_seed=736,8 \
    data=urmp \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    train.random_seed=172,828 \
    data=urmp \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \