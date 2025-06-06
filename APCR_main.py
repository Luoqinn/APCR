import time, argparse, datetime, torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from Anchor_based_Pairwise_Comparison import APC
from Position_Aware_List_Reranking import position_aware_list_reranking
from evaluation import evaluate_k


##################################################Parser#######################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='../llama3')
parser.add_argument('--cfm_rating_mat_name', type=str, default='datas/mf_rating_mat_1m.pt')
parser.add_argument('--cfm_label_name', type=str, default='datas/mf_label_mat_1m.pt')
parser.add_argument('--prompt_sequence1_file', type=str, default='datas/1m_anchor_prompt1.json')
parser.add_argument('--prompt_sequence2_file', type=str, default='datas/1m_anchor_prompt2.json')
parser.add_argument('--candidate_size', type=int, default=20)
parser.add_argument('--anchor_idx', type=int, default=0)
parser.add_argument('--decay_rate', type=float, default=0.88)
parser.add_argument('--device', type=str, default='cuda:1')
##################################################Parser#######################################################

if __name__ == "__main__":

    # load parameters
    args = parser.parse_args()
    cfm_rating_mat = torch.tensor(torch.load(args.cfm_rating_mat_name, weights_only=False))
    cfm_scores, cfm_idx = torch.topk(cfm_rating_mat, args.candidate_size, dim=1)
    label = torch.tensor(torch.load(args.cfm_label_name, weights_only=False))

    # APCR method
    print("Start Anchor-based Pairwise Comparison Reranking (APCR)!")
    start_time = time.time()
    apc = APC(args.model_name, args.prompt_sequence1_file, args.prompt_sequence2_file, args.candidate_size, args.anchor_idx, args.device)
    llm_preference_scores = apc.anchor_based_pairwise_comparison()
    final_reranked_item_list = position_aware_list_reranking(llm_preference_scores, args.decay_rate, cfm_rating_mat, args.candidate_size)
    end_time = time.time()
    print(f"Total time taken for APCR: {datetime.timedelta(seconds=end_time - start_time)}")

    # evaluation
    for k in [1, 10, 20]:
        ndcg, map = evaluate_k(torch.LongTensor(final_reranked_item_list)[:, :k], label, k)
        print(f"NDCG@{k}: {ndcg:.4f}\tMAP@{k}: {map:.4f}")




