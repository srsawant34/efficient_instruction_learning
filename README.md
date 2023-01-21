# Instruction Tuned Models are Quick Learners 

- This repo releases the implementation for our experiments of the research paper: "Instruction Tuned Models are Quick Learners"
- The experiments are run on [Tk-instruct-3B-model](https://huggingface.co/allenai/tk-instruct-3b-def-pos), which was finetuned on [data](https://github.com/allenai/natural-instructions).

## Requirements

The experiments and analysis are conducted on the following environment:

- CUDA (11.1)
- cuDNN (8.0)
- Pytorch (1.10.0)
- Transformers (4.18.0)
- DeepSpeed

For cloning the environment and install the required python libraries run the following command:

```bash
pip install -r requirements.txt
```

## Data

Our model are trained on the [Super-NaturalInstructions](https://github.com/allenai/natural-instructions) English-only tasks on 119 test tasks. The data splits can be created by running the following python script. 

```bash
python data_prep.py --num_examples 2 --ten True --onepercent True --hundred True --twohundred True --thousand True
```

## Running the experiment

To run the experiment, run the following the command:

```bash
sh scripts/master.sh -t task.txt -s twohundred
```

The above command will finetune the model(s) and evaluate it.
