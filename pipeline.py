import traceback

from task import build_tasks_from_file, MultiTask
from util import get_logger
from argparse import ArgumentParser
import torch

logger = get_logger(__name__)

arg_parser = ArgumentParser()

arg_parser.add_argument('-d', '--device',
                        type=int, default=0, help='GPU index')
# arg_parser.add_argument('-t', '--thread',
#                         type=int, default=5, help='Thread number')
arg_parser.add_argument('-c', '--config', help='Configuration file')

arg_parser.add_argument('-w', '--loss_weight', type=int, default=1)

args = arg_parser.parse_args()

torch.cuda.set_device(args.device)
# torch.set_num_threads(args.thread)
config_file = args.config
# _conf = Config.read(config_file)

loss_weight = args.loss_weight

tasks, conf, _ = build_tasks_from_file(config_file, options=None)
multitask = MultiTask(tasks, eval_freq=conf.training.eval_freq)

print ("The length of tasks is: ", len(tasks))

try:
    for step in range(1, conf.training.max_step + 1):
        multitask.step(loss_weight)
    # multitask.report()
        if step%conf.training.eval_freq==0:
            dev_score,test_score = multitask.get_best_score()
            print ("The best dev score of ref task: ", dev_score )
            print ("The best test score of ref task: ", test_score )
    # multitask.done()
except Exception:
    traceback.print_exc()
    # multitask.fail()
