import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str)
    arg.add_argument('-score', type=str, default='transe')
    arg.add_argument('-batch_size', type=int, default=512)
    arg.add_argument('-dim', type=int, default=512)
    arg.add_argument('-margin', type=float, default=5.0)
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str)
    arg.add_argument('-neg_num', type=int, default=128)
    arg.add_argument('-learning_rate', type=float, default=2e-5)
    arg.add_argument('-adv_temp', type=float, default=1.0)
    arg.add_argument('-seed', type=int, default=2022)
    arg.add_argument('-alpha', type=float, default=0.1)
    arg.add_argument('-beta1', type=float, default=0.1)
    arg.add_argument('-beta2', type=float, default=0.5)
    arg.add_argument('-drop', type=bool, default=False)
    arg.add_argument('-drop_p', type=float, default=0.5)
    arg.add_argument('-distmult_margin', type=float, default=20)
    # arg.add_argument('-rand_mix', type=bool, default=False)
    # arg.add_argument('-rand_p', type=float, default=0.1)
    arg.add_argument('-k', type=int, default=8)
    arg.add_argument('-inter_op', type=str, default="add")
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
