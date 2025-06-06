# APCR: Anchor-based Pairwise Comparison Reranking

## Requirements
- Python 3.7+
- PyTorch (>=2.0)
- Transformers
- tqdm
- NumPy

```bash
pip install torch transformers tqdm numpy
```

## Usage

### Command Line Arguments
| Argument | Type | Default | Description                                  |
|----------|------|---------|----------------------------------------------|
| `--model_name` | str | `../llama3` | Path to LLM model directory                  |
| `--cfm_rating_mat_name` | str | `datas/mf_rating_mat_1m.pt` | Collaborative filtering rating matrix        |
| `--cfm_label_name` | str | `datas/mf_label_mat_1m.pt` | Ground truth label matrix                    |
| `--prompt_sequence1_file` | str | `datas/1m_anchor_prompt1.json` | Prompt templates for first round comparison  |
| `--prompt_sequence2_file` | str | `datas/1m_anchor_prompt2.json` | Prompt templates for second round comparison |
| `--candidate_size` | int | 20 | Number of candidates to rerank               |
| `--anchor_idx` | int | 0 | Index of anchor item for comparisons         |
| `--decay_rate` | float | 0.88 | Position decay factor (0.8-0.9 recommended)  |
| `--device` | str | `cuda:1` | Computation device (`cpu`/`cuda`)            |

### Execution
```bash
python APCR_main.py \
  --model_name path/to/your/llm \
  --cfm_rating_mat_name path/to/rating_mat.pt \
  --cfm_label_name path/to/labels.pt \
  --candidate_size 20 \
  --decay_rate 0.88 \
  --device cuda:0
```

## Output
The script outputs:
1. APCR execution time
2. Evaluation metrics at K={1,10,20}:
   - HR@1 (Hit Rate) (when k=1)
   - NDCG@K (Normalized Discounted Cumulative Gain)
   - MAP@K (Mean Average Precision)

Example output:
```
Start Anchor-based Pairwise Comparison Reranking (APCR)!
Total time taken for APCR: 4:12:45.328
NDCG@1: 0.5457   MAP@1: 0.5457
NDCG@10: 0.4639  MAP@10: 0.4172
NDCG@20: 0.6065  MAP@20: 0.5511
```

## File Structure
```
├── APCR_main.py                
├── Anchor_based_Pairwise_Comparison.py  
├── Position_Aware_List_Reranking.py     
├── evaluation.py           
├── datas/                  # Sample data directory
│   ├── mf_rating_mat_1m.pt
│   ├── mf_label_mat_1m.pt
│   ├── 1m_anchor_prompt1.json
│   └── 1m_anchor_prompt2.json
```

## Customization
1. **Prompt Engineering**: Modify `prompt_sequence1_file` and `prompt_sequence2_file` to customize comparison prompts
2. **Decay Mechanism**: Adjust `--decay_rate` to control the influence of the positional contribution of the LLM suggested list on the original ranking scores by a CFM
3. **Candidate Pool**: Increase `--candidate_size` for broader reranking
4. **Anchor Selection**: Change `--anchor_idx` to fit prompts.
