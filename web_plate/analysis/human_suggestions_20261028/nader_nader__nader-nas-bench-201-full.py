import os
import warnings
import subprocess
import time
import argparse

from tools.utils import *
from nader import Nader


warnings.filterwarnings("ignore")


def print_and_save(text, file):
    print(text)
    with open(file, 'a') as f:
        f.write(text + '\n')
        f.flush()

blockmap={
    "resnet_basic":{777:"resnet_basic",888:"resnet_basic",999:"resnet_basic"},
    "nasbench_random":{
        777:"nasbench201_seed777",
        888:"resnet_basic",
        999:"nasbench201_seed777"
    }
}

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--max-iter',type=int,required=True)
    parser.add_argument('-d','--dataset',type=str,choices=['cifar10','cifar100','imagenet16-120'],required=True)
    parser.add_argument('-b','--base-block',type=str,default='resnet_basic',choices=['resnet_basic','convnext','nasbench_random'])
    parser.add_argument('-r','--research-team-name',type=str,default='nader',choices=['nader','nader_wor','nader_hc'])
    parser.add_argument('-c','--cluster',type=str,default='l40',choices=['l40','a800','4090d'])
    parser.add_argument('--seed',type=int,default=777,choices=[777,888,999])
    parser.add_argument('--width',default=5,type=int)
    parser.add_argument('--inspiration-retriever-mode',default='random',type=str,choices=['random','reflection'])
    parser.add_argument('--research-experience',default=None,type=str,choices=[None,'research_fail_240807_experience'])
    parser.add_argument('-p','--proposer-mode',type=str,default='llm')
    parser.add_argument('--log-dir',type=str,default='logs/nas-bench-201')
    parser.add_argument('--train-log-dir',type=str,default='output/nas-bench-201')

    args = parser.parse_args()
    cluster = args.cluster
    seed = args.seed
    max_iter = args.max_iter
    width = args.width
    dataset = args.dataset
    base_block=args.base_block
    research_team_name=args.research_team_name
    research_experience = args.research_experience
    inspiration_retriever_mode=args.inspiration_retriever_mode

    tag_prefix_base=f'{research_team_name}_{args.proposer_mode}_{base_block}_seed{seed}'
    log_dir = f'{args.log_dir}/{dataset}/{research_team_name}_{args.proposer_mode}_{base_block}_{inspiration_retriever_mode}_dfs-width{args.width}_run{args.max_iter}_seed{seed}'
    set_seed(seed)
    nader = Nader(
        base_block=blockmap[base_block][seed],
        dataset=f"nas-bench-201-{dataset}",
        database_dir='data/nas-bench-201',
        inspiration_retriever_mode=inspiration_retriever_mode,
        research_team_name=research_team_name,
        develop_team_name='try-fb',
        research_use_experience=research_experience,
        develop_use_experiecne='develop_allfailed_240709_experience',
        log_dir=log_dir,
        train_log_dir=f'{args.train_log_dir}/{dataset}',
        tag_prefix_base=tag_prefix_base,
        research_mode='dfs-one',
        proposer_mode=args.proposer_mode
    )

    best_test_acc,best_val_acc=-1,-1
    runs_log_path = os.path.join(log_dir,'log_runs.log')
    for iter in range(1,max_iter+1):
        start = time.time()
        print_and_save(f"Iter: {iter}/{max_iter}",runs_log_path)
        
        # create model
        nader.run_one_iter(iter=iter,width=width)
        path = os.path.join(log_dir,f"train_full_iter{iter}.sh")
        models = nader.generate_train_script(cluster=cluster,batch_path=path)
        print_and_save(f"Create models:{models}",runs_log_path)
        log_dirs = [os.path.join(nader.train_log_dir,model,'1') for model in models]
        e1 = time.time()
        
        process = subprocess.Popen(f'bash {path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while len(log_dirs)>0:
            ns = []
            status = True
            for d in log_dirs:
                path = os.path.join(d,'val_acc.txt')
                if not os.path.isfile(path):
                    n = 0
                else:
                    with open(path,'r') as f:
                        ds = f.readlines()
                        n = len(ds)-1
                ns.append(f"{n}/200")
                path = os.path.join(d,'train_status.txt')
                if not os.path.isfile(path):
                    status=False
            print_and_save(f"Process:{ns}",runs_log_path)
            if status:
                break
            time.sleep(60)
        print_and_save(f"Training finised!",runs_log_path)
        e2 = time.time()

        # summary
        block_list,test_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='test_acc',tag_prefix=tag_prefix_base,iter=iter,flush=True)
        block_list,val_acc = nader.research.mog_graph_manage.update_train_result(nader.train_log_dir,filename='val_acc',tag_prefix=tag_prefix_base,anno_name='mog_val',iter=iter,flush=True)
        best_test_acc = max(test_acc,best_test_acc)
        best_val_acc = max(val_acc,best_val_acc)
        print_and_save(f"Iter {iter}/{max_iter}, Val acc:{val_acc}, Test acc:{test_acc}, Best val acc{best_val_acc}, Best test acc:{best_test_acc}",runs_log_path)


