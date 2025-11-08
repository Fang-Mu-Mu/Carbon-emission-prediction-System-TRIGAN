from Margs_LSTM import args_parser
from Mget_data__LSTM import nn_seq, setup_seed
from Mmodels_LSTM import LSTM
from Muilts_LSTM import practice,practice_test
from Mget_data__LSTM import adj2coo
setup_seed(42)
def main():
    args = args_parser()
    train_loader,val_loader,test_loader, scaler = nn_seq(args)
    practice(args, train_loader)
    practice_test(args, test_loader,scaler)
if __name__ == '__main__':
    main()
