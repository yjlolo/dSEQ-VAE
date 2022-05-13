python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    data=urmp \
    train.random_seed=831 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    data=urmp \
    train.random_seed=736 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    data=urmp \
    train.random_seed=172 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    data=urmp \
    train.random_seed=277 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    data=urmp \
    train.random_seed=8 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \

python train.py -m \
    hydra/launcher=joblib \
    model=TsDsae \
    data=urmp \
    train.random_seed=828 \
    model.z_dim=8,16,32 \
    optim.optimizer.amsgrad=True,False \
    logging.media_log.reconstruction=False \
    logging.media_log.generation=False \
    logging.media_log.z_swap=False \
    logging.media_log.v_project=False \