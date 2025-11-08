from Margs_ELM import args_parser
from Mget_data_ELM import nn_seq, setup_seed
from Mmodels_ELM import ELM
from Muilts_ELM import practice,practice_test
from Mget_data_ELM import adj2coo
setup_seed(42)
def main():
    args = args_parser()
    train_loader,val_loader,test_loader, scaler = nn_seq(args)
    practice(args, train_loader)
    practice_test(args, test_loader,scaler)
if __name__ == '__main__':
    main()
