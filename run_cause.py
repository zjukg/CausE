import openke
import torch
from openke.config import Trainer, Tester
from openke.module.model import CDTransE, CDKGE, TransE, RotatE, DistMult, ComplEx
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from args import get_args

if __name__ == "__main__":
    # dataloader for training
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/{}/".format(args.dataset),
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )

    # dataloader for test
    test_dataloader = TestDataLoader(
        "./benchmarks/{}/".format(args.dataset), "link")

    # define the model
    if args.score == "transe":
        model_caus = TransE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=args.dim,
            p_norm=1,
            norm_flag=True
        )
        model_conf = TransE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=args.dim,
            p_norm=1,
            norm_flag=True
        )
    elif args.score == "rotate":
        model_caus = RotatE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=args.dim,
            epsilon=2.0
        )
        model_conf = RotatE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=args.dim,
            epsilon=2.0
        )
    elif args.score == "distmult":
        model_caus = DistMult(
	        ent_tot = train_dataloader.get_ent_tot(),
	        rel_tot = train_dataloader.get_rel_tot(),
	        dim=args.dim,
	        margin=args.margin,
	        epsilon=2.0
        )
        model_conf = DistMult(
	        ent_tot = train_dataloader.get_ent_tot(),
	        rel_tot = train_dataloader.get_rel_tot(),
	        dim=args.dim,
	        margin=args.margin,
	        epsilon=2.0
        )
    elif args.score == "complex":
        model_caus = ComplEx(
            ent_tot = train_dataloader.get_ent_tot(),
	        rel_tot = train_dataloader.get_rel_tot(),
	        dim = args.dim,
        )
        model_conf = ComplEx(
            ent_tot = train_dataloader.get_ent_tot(),
	        rel_tot = train_dataloader.get_rel_tot(),
	        dim = args.dim,
        )
    kge_score = CDKGE(
        model_caus=model_caus,
        model_conf=model_conf,
        margin=args.margin,
        alpha=0.5,
        # rand_p=args.rand_p,
        # k=args.k,
        inter_op=args.inter_op
    )
    # print(kge_score.inter_op, kge_score.inter_func)
    # define the loss function
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
        b1=args.beta1,
        b2=args.beta2
    )
    if args.score == "distmult":
        kge_score.margin_flag = False
        model.l3_regul_rate = 0.000005
    if args.score == "complex":
        kge_score.margin_flag = False
        model.regul_rate = 1.0
    # train the model
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        opt_method="Adam",
        use_gpu=True
    )
    trainer.run()
    kge_score.save_checkpoint('./checkpoint/{}.ckpt'.format(args.save))
    # test the model
    kge_score.load_checkpoint('./checkpoint/{}.ckpt'.format(args.save))
    if args.score == "distmult" or args.score == "complex":
        kge_score.margin_flag = False
        kge_score.model_caus.margin_flag = True
        kge_score.model_conf.margin_flag = True
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    kge_score.pred_type = "caus"
    tester.run_link_prediction(type_constrain=False)

