# Named Entity Recognition of Historical Text via Large Language Model


## Installation

1. Clone this repository
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Download HIPE-2022 dataset [here](https://github.com/hipe-eval/HIPE-2022-data) and unzip it in the dataset folder

## Usage

### Generating NER Predictions

To generate NER predictions, use the following command:

```bash
python -m src.main {dataset_name} {dataset_lang} {split} 0 coarse {prompt_method} {APIkey} deepseek-chat {subfolder}
```

#### Parameters

- **dataset_name**: The name of the dataset to process
- **dataset_lang**: The language of the dataset
- **split**: Data split to use (`train`, `dev`, or `test`)
- **prompt_method**: The prompting method to use (see available methods below)
- **APIkey**: Your DeepSeek API key (you need to pay for it yourself)
- **subfolder**: Folder name where predictions will be saved

#### Available Prompt Methods

| Method | Description                                                |
|--------|------------------------------------------------------------|
| b | Baseline                                                   |
| r_m1 | Prompt with 1 randomly retreived example                   |
| r_m3 | Prompt with 3 randomly retreived example                   |
| r_m5 | Prompt with 5 randomly retreived example                   |
| s_overlap_1 | Prompt with 1 example retrieved using lexical similarity   |
| s_overlap_3 | Prompt with 3 example retrieved using lexical similarity   |
| s_overlap_5 | Prompt with 5 example retrieved using lexical similarity   |
| s_embedding_1 | Prompt with 1 example retrieved using embedding similarity |
| s_embedding_3 | Prompt with 3 example retrieved using embedding similarity |
| s_embedding_5 | Prompt with 5 example retrieved using embedding similarity |

#### Supported Datasets and Languages

| Dataset | Supported Languages |
|---------|-------------------|
| ajmc | de, en, fr |
| hipe2020 | de, en, fr |
| letemps | fr |
| newseye | de, fi, fr, sv |
| sonar | de |
| topres19th | en |

### Evaluation

To evaluate the predictions, use the following command:

```bash
python -m src.evaluation {dataset_name} {dataset_lang} {split} {prompt_method} deepseek-chat 0 {subfolder} nerc_coarse
```

#### Parameters

All parameters are the same as for prediction generation

## Example Usage

### Generating Predictions
```bash
python -m src.main ajmc en train 0 coarse b your_api_key_here deepseek-chat results_folder
```

### Running Evaluation
```bash
python -m src.evaluation ajmc en train b deepseek-chat 0 results_folder nerc_coarse
```
## Acknowledgements
The evaluation code in this project is partially adapted from [HIPE-scorer](https://github.com/hipe-eval/HIPE-scorer).
