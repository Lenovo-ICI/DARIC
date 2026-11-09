import json
from metrics import compute_f1, compute_rl, code_sim_score

class DataLoader:
    def __init__(self, dataset: str, config: dict, k: int=10):
        self.dataset = dataset
        self.config = config
        self.max_tokens = self.config["max_tokens"]
        self.system_prompt_template = self.config["system_prompt"]
        self.user_prompt_template = self.config["user_prompt"]
        self.k = k
        if self.config["metric"] == "f1":
            self.metric = compute_f1
        elif self.config["metric"] == "rl":
            self.metric = compute_rl
        elif self.config["metric"] == "edit_sim":
            self.metric = code_sim_score
        else:
            raise ValueError(f"Unknown metric: {self.config['metric']}")
        print(f"Initialized DataLoader for {dataset} with metric {self.config['metric']}")
        with open(self.config["path"], "r") as f:
            eval_dataset = json.load(f)
        self.data = []
        for ex in eval_dataset:
            answers = ex["answers"]
            if not isinstance(answers[0], str):
                answers = [answer[0] for answer in answers]
            
            if dataset in ["wikimqa", "musique", "narrativeqa"]:
                doc_prompts, q_prompt = self.build_qa_prompt(ex)
            elif dataset in ["samsum"]:
                doc_prompts, q_prompt = self.build_fewshot_prompt(ex)
            elif dataset in ["multinews"]:
                doc_prompts, q_prompt = self.build_summary_prompt(ex)
            self.data.append((doc_prompts, q_prompt, answers))

    def normalize_question(self, question):
        if not question.endswith("?"):
            question = question + "?"

        return question[0].lower() + question[1:]

    def build_qa_prompt(self, example):
        q = self.normalize_question(example["question"])
        doc_prompts = [
            f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"][: self.k]
        ]
        q_prompt = f"{self.user_prompt_template}{q}\nAnswer:"
        return doc_prompts, q_prompt

    def build_fewshot_prompt(self, example):
        q = "\n\n" + example["question"]
        doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]][: self.k]
        q_prompt = f"{q}"
        return doc_prompts, q_prompt
    
    def build_summary_prompt(self, example):
        doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]][: self.k]
        q_prompt = f"{self.user_prompt_template}"
        return doc_prompts, q_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]