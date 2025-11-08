from Margs import args_parser
from Mget_data11 import nn_seq, setup_seed
from Mmodels import SAEG_Net
from Muilts import test, train,practice,practice_test
from Mget_data import adj2coo
setup_seed(42)

def main():
    args = args_parser()
    Dtr, Val, Dte, scaler, edge_index = nn_seq(args)

    practice(args, Dtr, edge_index)
    practice_test(args, Dte,scaler, edge_index)
if __name__ == '__main__':
    main()