
import sys
import os
import unetParams as params
from Baseline import Baseline
from datafunctions import parse_args


'''
Main script to be run

Calls:
training: python NetworkRun train
testing: python NetworkRun test
MC Dropout testing: python NetworkRun droptest

types refinetraindata and refinetrain are available, but our experiments did not improve using these models.
They were atempts of active learning models.
'''


def main(argv):
    training = parse_args(argv, params)
    os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
    os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    model = Baseline(params=params)
    if training == 0:
        #train the network
        try:
            model.train(type_train=0)
        except ValueError as error:
            print(error, file=sys.stderr)
    elif training == 1:
        #test the Network
        try:
            model.test()
        except ValueError as error:
            print(error, file=sys.stderr)
    elif training == 2:
        #Test the network with MCDropout
        try:
            model.predict_drop()
        except ValueError as error:
            print(error, file=sys.stderr)
    elif training == 3:
        #Infere nos dados de treinamento (treinamento e validacao) para os refinar, em modelo de active-learning
        try:#Test train and validation data to refine them based on MCDropout.
            model.refineTrainDataset()
        except ValueError as error:
            print(error, file=sys.stderr)
    elif training == 4:
        #Continue training with refined training and validation patches (after 3)
        try:
            model.train(type_train=1)
        except ValueError as error:
            print(error, file=sys.stderr)
    else:
        raise ValueError("Incorrect model function definition. Possible arguments: train, test, droptest, refinetraindata, refinetrain")       


if __name__ == "__main__":
    main(sys.argv)

