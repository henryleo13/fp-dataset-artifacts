from typing import Dict
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainerControl, TrainerState, TrainingArguments, HfArgumentParser, TrainerCallback
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, compute_accuracy_hans, compute_accuracy_and_c_above90
import os, time, json, torch
from transformers.trainer_utils import speed_metrics
import torch.nn.functional as F

NUM_PREPROCESSING_WORKERS = 2

# Create a subclass of Trainer and modify prediction_step
# to encode 2 as 1, 1 as 1 and 0 as 0
class MyTrainer(Trainer):
    def __init__(self, biased_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if biased_model:
            self.biased_model = biased_model
            self.biased_model = self.biased_model.to(self.args.device)

    #def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and returns metrics.
        Will return a namedtuple with the following keys:
        - loss: scalar
        - metrics: dict of (string, float) with the metric name as key and the metric's value as the value.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. Otherwise, it will default to
                :obj:`self.eval_dataset`.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of the model (if it's a dictionary) that should be ignored when gathering
                predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval".
        """
        """ eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, output.num_samples * 1000 / total_batch_size))
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        #self.log_metrics('eval', output.metrics)
        #self.save_metrics('eval', output.metrics)
        return output.metrics """



class DebiasTrainer(Trainer):
    def __init__(self, biased_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if biased_model:
            self.biased_model = biased_model
            self.biased_model = self.biased_model.to(self.args.device)

        # Freeze biased_model
        for param in self.biased_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        labels = inputs['labels']
        outputs = model(**inputs)
        output_logits = outputs[1]

        biased_outputs = self.biased_model(**inputs)
        biased_logits = biased_outputs[1]

        PoE_Loss = self.ensemble_loss(output_logits, biased_logits, labels)

        return (PoE_Loss, outputs) if return_outputs else PoE_Loss
    
    def ensemble_loss(self, output_logits, biased_logits, labels):

        # Compute Product-of-Experts loss
        return F.cross_entropy(output_logits + biased_logits, labels)  
    
class AccuracyCallback(TrainerCallback):
    def __init__(self, eval_steps, logging_steps):
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps

    def on_train_begin(self, args, stage, control, **kwargs):
        print("Starting Training!")

    #def on_step_end(self, args, state, control, **kwargs):
    #    if state.global_step % self.eval_steps == 0:
    #        control.should_evaluate = True

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print("Evaluation results:", metrics)
        #control.should_save = True
        return control
    



def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--train_size', type=int, default=200,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--eval_size', type=int, default=50,
                      help='Limit the number of examples to evaluate on.')
    
    # Add arguments for biased model
    argp.add_argument('--biased_model', type=str, default=None,
                        help="""This argument specifies the biased modele.
            This should be a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")



    training_args, args = argp.parse_args_into_dataclasses()

   
    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'

        # if args.output_dir == "./biased_model/" then split the dataset into train and eval, 2k in train and 0.5k in eval
        if training_args.output_dir == "./biased_model/":
            # set seed
            torch.manual_seed(42)
            dataset = dataset['train'].train_test_split(test_size=args.eval_size, shuffle=True)
            dataset['train'] = dataset['train'].train_test_split(train_size=args.train_size, shuffle=True)['train']
            eval_split = 'test'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load the biased model and tokenizer from the specified pretrained model/checkpoint
    biased_model = None
    if args.biased_model is not None:
        biased_model = model_class.from_pretrained(args.biased_model, **task_kwargs)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    dataset_type1 = ['./datasets/multinli_1.0/multinli_1.0_train.jsonl',
                     './datasets/Breaking_NLI-master/data/dataset.jsonl', 
                    './datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl',
                    './datasets/multinli_1.0/multinli_1.0_dev_mismatched.jsonl']
    dataset_type2 = ['./datasets/heuristics_evaluation_set.jsonl',
                    './datasets/heuristics_evaluation_set_2000.jsonl',
                    './datasets/heuristics_evaluation_set_500.jsonl']
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        
        if args.dataset in dataset_type1:
            # Chnage key value of sentence1 and sentence2 to premise and hypothesis
            # map label to 0, 1, 2
            train_dataset = train_dataset.map(
                lambda ex: {'premise': ex['sentence1'], 'hypothesis': ex['sentence2'], 'label': 0 if ex['gold_label'] == 'entailment' else 1 if ex['gold_label'] == 'neutral' else 2},
                remove_columns=train_dataset.column_names
            )
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        if args.dataset in dataset_type1:
            # Chnage key value of sentence1 and sentence2 to premise and hypothesis
            # map label to 0, 1, 2
            eval_dataset = eval_dataset.map(
                lambda ex: {'premise': ex['sentence1'], 'hypothesis': ex['sentence2'], 'label': 0 if ex['gold_label'] == 'entailment' else 1 if ex['gold_label'] == 'neutral' else 2},
                remove_columns=eval_dataset.column_names
            )
        elif args.dataset in dataset_type2:
            eval_dataset = eval_dataset.map(
                lambda ex: {'premise': ex['sentence1'], 'hypothesis': ex['sentence2'], 'label': 0 if ex['gold_label'] == 'entailment' else 1},
                remove_columns=eval_dataset.column_names
            )

        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration

    if args.biased_model is not None:
        trainer_class = DebiasTrainer
    else:
        #trainer_class = Trainer
        trainer_class = MyTrainer


    #if training_args.output_dir == "./student_model/":

    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        if training_args.output_dir == "./biased_model/":
            compute_metrics = compute_accuracy_and_c_above90
        else: 
            compute_metrics = compute_accuracy_hans if args.dataset in dataset_type2 else compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    callbacks = None
    step_size = int(512 / (training_args.per_device_train_batch_size))
    if training_args.do_eval:
        #callbacks = [AccuracyCallback(eval_steps=200, logging_steps=200)]
        training_args.evaluation_strategy="steps"
        training_args.logging_strategy = "steps"
        training_args.save_strategy = "epoch"
        training_args.eval_steps = step_size
        training_args.logging_steps = step_size
    
    trainer = trainer_class(
        model=model,
        biased_model=biased_model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
        callbacks = callbacks
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    
                    pred = eval_predictions.predictions[i].argmax()
                    if args.dataset in dataset_type2:
                        pred = 1 if pred == 2 else 0
                    example_with_prediction['predicted_label'] = int(pred)
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
